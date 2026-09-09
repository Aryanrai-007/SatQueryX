from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO
from typing import Any

import folium
import numpy as np
import rasterio
import requests
from folium.plugins import Draw
from rasterio.io import MemoryFile
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from streamlit_folium import st_folium

STAC_URL = "https://earth-search.aws.element84.com/v1"
S2_COLLECTION = "sentinel-2-l2a"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
APP_UA = "SatQueryX/0.1 (+https://github.com/Aryanrai-007/SatQueryX)"


def build_aoi_map(center: tuple[float, float], zoom: int = 11, key: str = "satqueryx_aoi_map"):
    m = folium.Map(
        location=list(center),
        zoom_start=zoom,
        tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="© OpenStreetMap contributors",
        control_scale=True,
        prefer_canvas=True,
    )
    Draw(
        export=False,
        position="topleft",
        draw_options={
            "polyline": False,
            "polygon": True,
            "rectangle": True,
            "circle": False,
            "circlemarker": False,
            "marker": False,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(m)
    return st_folium(
        m,
        height=520,
        use_container_width=True,
        key=key,
        returned_objects=["last_active_drawing", "all_drawings", "center", "zoom"],
    )


def geometry_from_drawing(drawing: dict[str, Any] | None) -> dict[str, Any] | None:
    if not drawing:
        return None
    geometry = drawing.get("geometry") if isinstance(drawing, dict) else None
    return geometry if geometry and geometry.get("type") in {"Polygon", "MultiPolygon"} else None


def geometry_bbox(geometry: dict[str, Any]) -> tuple[float, float, float, float]:
    coords = geometry.get("coordinates", [])

    def points(value):
        if value and isinstance(value[0], (int, float)):
            yield value
        else:
            for child in value:
                yield from points(child)

    pts = list(points(coords))
    if not pts:
        raise ValueError("The selected area contains no coordinates.")
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    minx, miny, maxx, maxy = bbox
    return ((miny + maxy) / 2.0, (minx + maxx) / 2.0)


def reverse_geocode(lat: float, lon: float) -> str:
    response = requests.get(
        NOMINATIM_URL,
        params={"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 10},
        headers={"User-Agent": APP_UA},
        timeout=15,
    )
    response.raise_for_status()
    return str(response.json().get("display_name") or "Location name unavailable")


def _stac_error(response: requests.Response) -> str:
    try:
        payload = response.json()
        detail = payload.get("description") or payload.get("detail") or payload.get("title")
        if detail:
            return str(detail)
    except ValueError:
        pass
    text = response.text.strip()
    return text[:500] if text else f"HTTP {response.status_code}"


def search_sentinel2(
    bbox: tuple[float, float, float, float],
    start_date: date,
    end_date: date,
    max_cloud: float = 15.0,
    limit: int = 12,
) -> list[dict[str, Any]]:
    if start_date > end_date:
        raise ValueError("Search start date must be on or before the search end date.")

    url = f"{STAC_URL}/search"
    headers = {"User-Agent": APP_UA, "Accept": "application/geo+json"}
    # STAC datetime is RFC3339, not date-only ISO-8601.
    start_ts = f"{start_date.isoformat()}T00:00:00Z"
    end_ts = f"{end_date.isoformat()}T23:59:59Z"
    base_payload = {
        "collections": [S2_COLLECTION],
        "bbox": list(bbox),
        "datetime": f"{start_ts}/{end_ts}",
        "limit": int(limit),
    }
    filtered_payload = {
        **base_payload,
        "query": {"eo:cloud_cover": {"lte": float(max_cloud)}},
    }
    response = requests.post(url, json=filtered_payload, headers=headers, timeout=30)

    if response.status_code == 400:
        fallback = requests.post(url, json=base_payload, headers=headers, timeout=30)
        if fallback.ok:
            features = fallback.json().get("features", [])
            return [
                feature
                for feature in features
                if float(feature.get("properties", {}).get("eo:cloud_cover", 101.0)) <= float(max_cloud)
            ]
        raise RuntimeError(f"Earth Search rejected the AOI search ({fallback.status_code}): {_stac_error(fallback)}")

    if not response.ok:
        raise RuntimeError(f"Earth Search rejected the AOI search ({response.status_code}): {_stac_error(response)}")
    return response.json().get("features", [])


def _asset_href(item: dict[str, Any], *names: str) -> str:
    assets = item.get("assets", {})
    lowered = {str(k).lower(): v for k, v in assets.items()}
    for name in names:
        if name.lower() in lowered and lowered[name.lower()].get("href"):
            return lowered[name.lower()]["href"]
    for key, asset in assets.items():
        key_l = str(key).lower()
        if any(name.lower() in key_l for name in names) and asset.get("href"):
            return asset["href"]
    raise KeyError(f"Could not find any of the requested assets: {names}")


def _asset_available(item: dict[str, Any]) -> bool:
    """Probe a representative 10 m COG before selecting a scene.

    Earth Search can expose metadata before every referenced COG is reachable.
    A lightweight HEAD/range probe lets SatQueryX skip such scenes instead of
    failing the entire AOI workflow on a 404 during rasterio's first read.
    """
    try:
        href = _asset_href(item, "red", "b04")
    except KeyError:
        return False

    try:
        response = requests.head(
            href,
            headers={"User-Agent": APP_UA},
            allow_redirects=True,
            timeout=12,
        )
        if response.status_code in {200, 206}:
            return True
        if response.status_code not in {403, 405}:
            return False
    except requests.RequestException:
        pass

    try:
        response = requests.get(
            href,
            headers={"User-Agent": APP_UA, "Range": "bytes=0-0"},
            stream=True,
            allow_redirects=True,
            timeout=12,
        )
        ok = response.status_code in {200, 206}
        response.close()
        return ok
    except requests.RequestException:
        return False


def _clip_asset(url: str, bbox: tuple[float, float, float, float], max_dimension: int = 4096):
    # Explicit /vsicurl/ makes the intended remote, range-readable COG path
    # unambiguous to GDAL. SERIAL multi-range mode is conservative for S3 and
    # avoids relying on a proxy to support multipart Range responses.
    vsi_url = url if url.startswith("/vsicurl/") else f"/vsicurl/{url}"
    with rasterio.Env(
        GDAL_HTTP_USERAGENT=APP_UA,
        GDAL_HTTP_MULTIRANGE="SERIAL",
        GDAL_HTTP_MAX_RETRY="2",
        GDAL_HTTP_RETRY_CODES="429,500,502,503,504",
        CPL_VSIL_CURL_USE_HEAD="NO",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    ):
        with rasterio.open(vsi_url) as src:
            left, bottom, right, top = transform_bounds(
                "EPSG:4326", src.crs, *bbox, densify_pts=21
            )
            left, right = max(left, src.bounds.left), min(right, src.bounds.right)
            bottom, top = max(bottom, src.bounds.bottom), min(top, src.bounds.top)
            if left >= right or bottom >= top:
                raise ValueError("The selected AOI does not overlap the selected satellite scene.")
            window = from_bounds(left, bottom, right, top, transform=src.transform).round_offsets().round_lengths()
            width, height = max(1, int(window.width)), max(1, int(window.height))
            scale = min(1.0, max_dimension / max(width, height))
            out_width, out_height = max(1, int(width * scale)), max(1, int(height * scale))
            data = src.read(
                1,
                window=window,
                out_shape=(out_height, out_width),
                resampling=rasterio.enums.Resampling.bilinear,
            )
            transform = src.window_transform(window)
            if scale != 1.0:
                transform = transform * rasterio.Affine.scale(width / out_width, height / out_height)
            return data.astype(np.float32), transform, src.crs


def fetch_sentinel2_snippet(
    item: dict[str, Any],
    bbox: tuple[float, float, float, float],
    max_dimension: int = 4096,
) -> tuple[bytes, dict[str, Any]]:
    bands = [
        ("blue", ("blue", "b02")),
        ("green", ("green", "b03")),
        ("red", ("red", "b04")),
        ("nir", ("nir", "b08")),
    ]
    arrays = []
    transform = None
    crs = None
    for _, aliases in bands:
        href = _asset_href(item, *aliases)
        array, item_transform, item_crs = _clip_asset(href, bbox, max_dimension=max_dimension)
        if transform is None:
            transform, crs = item_transform, item_crs
        elif array.shape != arrays[0].shape:
            raise ValueError("Sentinel-2 band windows have incompatible shapes.")
        arrays.append(array)

    stack = np.stack(arrays).astype(np.uint16)
    profile = {
        "driver": "GTiff",
        "height": stack.shape[1],
        "width": stack.shape[2],
        "count": 4,
        "dtype": "uint16",
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "BIGTIFF": "IF_SAFER",
    }
    output = BytesIO()
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(stack)
            dst.set_band_description(1, "Blue (B02)")
            dst.set_band_description(2, "Green (B03)")
            dst.set_band_description(3, "Red (B04)")
            dst.set_band_description(4, "NIR (B08)")
        output.write(mem.read())

    props = item.get("properties", {})
    metadata = {
        "scene_id": item.get("id", "unknown"),
        "datetime": props.get("datetime") or props.get("start_datetime"),
        "cloud_cover": props.get("eo:cloud_cover"),
        "collection": item.get("collection") or S2_COLLECTION,
        "source": "Element84 Earth Search / AWS Open Data",
        "bbox": bbox,
        "crs": str(crs) if crs else None,
        "bands": ["B02", "B03", "B04", "B08"],
        "platform": props.get("platform") or "Sentinel-2",
        "instruments": props.get("instruments") or ["MSI"],
        "mgrs_tile": props.get("s2:mgrs_tile"),
        "epsg": props.get("proj:epsg"),
    }
    return output.getvalue(), metadata


def select_items(
    features: list[dict[str, Any]],
    count: int = 2,
    min_gap_days: int = 14,
) -> list[dict[str, Any]]:
    if not features:
        return []

    parsed = [
        (f.get("properties", {}).get("datetime"), f)
        for f in features
        if f.get("properties", {}).get("datetime")
    ]
    parsed.sort(key=lambda x: x[0], reverse=True)

    # Prefer scenes whose representative COG is actually reachable. If a test
    # fixture has no assets, preserve the old selection behavior.
    usable = []
    for value, feature in parsed:
        assets = feature.get("assets") or {}
        if assets and not _asset_available(feature):
            continue
        usable.append((value, feature))

    if not usable:
        return []

    selected = [usable[0][1]]
    first_date = date.fromisoformat(usable[0][0][:10])
    if count == 1:
        return selected

    for value, feature in usable[1:]:
        if abs((first_date - date.fromisoformat(value[:10])).days) >= min_gap_days:
            selected.append(feature)
            break
    return selected[:count]


def default_dates(days: int = 365) -> tuple[date, date]:
    end = date.today()
    return end - timedelta(days=days), end
