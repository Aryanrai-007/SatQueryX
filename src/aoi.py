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
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
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
    folium.LayerControl(collapsed=True).add_to(m)
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
    if not geometry or geometry.get("type") not in {"Polygon", "MultiPolygon"}:
        return None
    return geometry


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
    data = response.json()
    return str(data.get("display_name") or "Location name unavailable")


def search_sentinel2(
    bbox: tuple[float, float, float, float],
    start_date: date,
    end_date: date,
    max_cloud: float = 15.0,
    limit: int = 12,
) -> list[dict[str, Any]]:
    payload = {
        "collections": [S2_COLLECTION],
        "bbox": list(bbox),
        "datetime": f"{start_date.isoformat()}/{end_date.isoformat()}",
        "query": {"eo:cloud_cover": {"lte": float(max_cloud)}},
        "limit": int(limit),
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
    }
    response = requests.post(
        f"{STAC_URL}/search",
        json=payload,
        headers={"User-Agent": APP_UA, "Accept": "application/geo+json"},
        timeout=30,
    )
    response.raise_for_status()
    features = response.json().get("features", [])
    return features


def _asset_href(item: dict[str, Any], *names: str) -> str:
    assets = item.get("assets", {})
    lowered = {str(k).lower(): v for k, v in assets.items()}
    for name in names:
        if name.lower() in lowered:
            href = lowered[name.lower()].get("href")
            if href:
                return href
    for key, asset in assets.items():
        key_l = str(key).lower()
        if any(name.lower() in key_l for name in names):
            href = asset.get("href")
            if href:
                return href
    raise KeyError(f"Could not find any of the requested assets: {names}")


def _clip_asset(url: str, bbox: tuple[float, float, float, float], max_dimension: int = 4096):
    with rasterio.Env(
        GDAL_HTTP_USERAGENT=APP_UA,
        GDAL_HTTP_MULTIRANGE="YES",
        CPL_VSIL_CURL_USE_HEAD="NO",
    ):
        with rasterio.open(url) as src:
            left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=21)
            left = max(left, src.bounds.left)
            right = min(right, src.bounds.right)
            bottom = max(bottom, src.bounds.bottom)
            top = min(top, src.bounds.top)
            if left >= right or bottom >= top:
                raise ValueError("The selected AOI does not overlap the selected satellite scene.")
            window = from_bounds(left, bottom, right, top, transform=src.transform)
            window = window.round_offsets().round_lengths()
            width = max(1, int(window.width))
            height = max(1, int(window.height))
            scale = min(1.0, max_dimension / max(width, height))
            out_width = max(1, int(width * scale))
            out_height = max(1, int(height * scale))
            data = src.read(1, window=window, out_shape=(out_height, out_width), resampling=rasterio.enums.Resampling.bilinear)
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
        "tiled": True,
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
    }
    return output.getvalue(), metadata


def select_items(features: list[dict[str, Any]], count: int = 2, min_gap_days: int = 14) -> list[dict[str, Any]]:
    if not features:
        return []
    parsed = []
    for feature in features:
        value = feature.get("properties", {}).get("datetime")
        if value:
            parsed.append((value, feature))
    parsed.sort(key=lambda x: x[0], reverse=True)
    selected = [parsed[0][1]]
    first_dt = parsed[0][0][:10]
    first_date = date.fromisoformat(first_dt)
    for _, feature in parsed[1:]:
        dt = feature.get("properties", {}).get("datetime", "")[:10]
        if not dt:
            continue
        if abs((first_date - date.fromisoformat(dt)).days) >= min_gap_days:
            selected.append(feature)
            break
    return selected[:count]


def default_dates(days: int = 365) -> tuple[date, date]:
    end = date.today()
    return end - timedelta(days=days), end
