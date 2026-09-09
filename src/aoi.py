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
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from rasterio.enums import Resampling
from streamlit_folium import st_folium

STAC_URL = "https://earth-search.aws.element84.com/v1"
S2_COLLECTION = "sentinel-2-l2a"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
APP_UA = "SatQueryX/0.1 (+https://github.com/Aryanrai-007/SatQueryX)"


def build_aoi_map(center: tuple[float, float], zoom: int = 11, key: str = "satqueryx_aoi_map"):
    m = folium.Map(
        location=list(center),
        zoom_start=zoom,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )

    folium.TileLayer(
        tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="© OpenStreetMap contributors",
        name="🗺️ Street map",
        overlay=False,
        control=True,
        max_zoom=19,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="🛰️ Satellite imagery",
        overlay=False,
        control=True,
        max_zoom=19,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="© OpenTopoMap contributors",
        name="⛰️ Topographic",
        overlay=False,
        control=True,
        max_zoom=17,
    ).add_to(m)

    folium.LayerControl(position="topright", collapsed=False).add_to(m)

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
    return response.text[:500] or f"HTTP {response.status_code}"


def search_sentinel2(
    bbox: tuple[float, float, float, float],
    start: date,
    end: date,
    max_cloud: float = 20.0,
) -> list[dict[str, Any]]:
    if start > end:
        raise ValueError("Start date must be before end date.")
    start_rfc3339 = f"{start.isoformat()}T00:00:00Z"
    end_rfc3339 = f"{end.isoformat()}T23:59:59Z"
    payload = {
        "collections": [S2_COLLECTION],
        "bbox": list(bbox),
        "datetime": f"{start_rfc3339}/{end_rfc3339}",
        "limit": 50,
        "query": {"eo:cloud_cover": {"lte": float(max_cloud)}},
    }
    response = requests.post(
        f"{STAC_URL}/search",
        json=payload,
        headers={"User-Agent": APP_UA},
        timeout=25,
    )
    if response.status_code == 400:
        payload.pop("query", None)
        response = requests.post(
            f"{STAC_URL}/search",
            json=payload,
            headers={"User-Agent": APP_UA},
            timeout=25,
        )
    if not response.ok:
        raise RuntimeError(
            f"Earth Search rejected the AOI search ({response.status_code}): {_stac_error(response)}"
        )
    features = response.json().get("features", [])
    return [
        item
        for item in features
        if float(item.get("properties", {}).get("eo:cloud_cover", 999)) <= max_cloud
    ]


def _asset_href(item: dict[str, Any], *names: str) -> str | None:
    assets = item.get("assets", {})
    for name in names:
        href = assets.get(name, {}).get("href")
        if href:
            return href
    return None


def _asset_available(item: dict[str, Any]) -> bool:
    href = _asset_href(item, "red", "B04", "visual")
    if not href:
        return True
    try:
        response = requests.head(
            href,
            headers={"User-Agent": APP_UA},
            timeout=(5, 8),
            allow_redirects=True,
        )
        if response.status_code in {200, 206}:
            return True
        if response.status_code in {403, 405, 501}:
            response = requests.get(
                href,
                headers={"User-Agent": APP_UA, "Range": "bytes=0-0"},
                timeout=(5, 8),
                stream=True,
            )
            return response.status_code in {200, 206}
    except requests.RequestException:
        return False
    return False


def _clip_asset(href: str, bbox: tuple[float, float, float, float]) -> np.ndarray:
    """Read an AOI from a remote Sentinel-2 COG in a stable WGS84 grid.

    Some remote assets expose incomplete/native transform metadata through the
    HTTP/COG path. WarpedVRT gives rasterio an explicit destination transform,
    so from_bounds never receives a missing transform and the AOI remains
    geospatially correct.
    """
    with rasterio.Env(
        GDAL_HTTP_USERAGENT=APP_UA,
        GDAL_HTTP_MULTIRANGE="SERIAL",
        GDAL_HTTP_MAX_RETRY="2",
        GDAL_HTTP_RETRY_CODES="429,500,502,503,504",
        CPL_VSIL_CURL_USE_HEAD="NO",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    ):
        with rasterio.open(f"/vsicurl/{href}") as ds:
            if ds.crs is None:
                raise RuntimeError("Sentinel-2 asset has no CRS metadata; cannot safely clip the AOI.")

            # First transform the AOI into the source CRS only to test overlap.
            source_bbox = transform_bounds(
                "EPSG:4326", ds.crs, *bbox, densify_pts=21
            )
            if (
                source_bbox[2] <= ds.bounds.left
                or source_bbox[0] >= ds.bounds.right
                or source_bbox[3] <= ds.bounds.bottom
                or source_bbox[1] >= ds.bounds.top
            ):
                raise ValueError("Selected AOI does not overlap the satellite raster.")

            # Reproject on-the-fly into WGS84. This supplies a concrete affine
            # transform even when the source COG's transform is not exposed
            # correctly by the remote filesystem layer.
            vrt_width = max(1, int(round((bbox[2] - bbox[0]) * 111_320 / 10.0)))
            vrt_height = max(1, int(round((bbox[3] - bbox[1]) * 110_540 / 10.0)))
            vrt_width = min(vrt_width, 2048)
            vrt_height = min(vrt_height, 2048)
            with WarpedVRT(
                ds,
                crs="EPSG:4326",
                transform=rasterio.transform.from_bounds(
                    *bbox, vrt_width, vrt_height
                ),
                width=vrt_width,
                height=vrt_height,
                resampling=Resampling.bilinear,
            ) as vrt:
                window = from_bounds(*bbox, transform=vrt.transform)
                window = window.round_offsets().round_lengths()
                window = window.intersection(
                    rasterio.windows.Window(0, 0, vrt.width, vrt.height)
                )
                if window.width <= 0 or window.height <= 0:
                    raise ValueError("Selected AOI does not overlap the satellite raster.")
                return vrt.read(1, window=window, out_dtype="float32")


def fetch_sentinel2_snippet(
    item: dict[str, Any], bbox: tuple[float, float, float, float]
) -> tuple[bytes, dict[str, Any]]:
    hrefs = {
        "B02": _asset_href(item, "blue", "B02"),
        "B03": _asset_href(item, "green", "B03"),
        "B04": _asset_href(item, "red", "B04"),
        "B08": _asset_href(item, "nir", "B08"),
    }
    missing = [band for band, href in hrefs.items() if not href]
    if missing:
        raise RuntimeError(
            f"Selected Sentinel-2 scene is missing required COG assets: {', '.join(missing)}"
        )

    arrays = [_clip_asset(hrefs[band], bbox) for band in ("B02", "B03", "B04", "B08")]
    shapes = {a.shape for a in arrays}
    if len(shapes) != 1:
        raise RuntimeError("Sentinel-2 band windows do not align exactly for the selected AOI.")

    profile = {
        "driver": "GTiff",
        "height": arrays[0].shape[0],
        "width": arrays[0].shape[1],
        "count": 4,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": rasterio.transform.from_bounds(
            *bbox, arrays[0].shape[1], arrays[0].shape[0]
        ),
        "compress": "deflate",
    }
    out = BytesIO()
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            for idx, arr in enumerate(arrays, start=1):
                dst.write(arr, idx)
        out.write(mem.read())

    props = item.get("properties", {})
    metadata = {
        "scene_id": item.get("id"),
        "datetime": props.get("datetime"),
        "cloud_cover": props.get("eo:cloud_cover"),
        "collection": item.get("collection", S2_COLLECTION),
        "source": "Element84 Earth Search / Sentinel-2 L2A COG",
        "bbox": list(bbox),
        "crs": "EPSG:4326",
        "bands": ["B02", "B03", "B04", "B08"],
        "platform": props.get("platform"),
        "instruments": props.get("instruments", []),
        "mgrs_tile": props.get("s2:mgrs_tile"),
        "epsg": props.get("proj:epsg"),
    }
    return out.getvalue(), metadata


def select_items(
    features: list[dict[str, Any]], count: int = 1, min_gap_days: int = 14
) -> list[dict[str, Any]]:
    ordered = sorted(
        features,
        key=lambda item: item.get("properties", {}).get("datetime") or "",
        reverse=True,
    )
    reachable = [item for item in ordered if _asset_available(item)]
    if not reachable:
        reachable = ordered
    if count == 1:
        return reachable[:1]
    selected: list[dict[str, Any]] = []
    for item in reachable:
        if not selected:
            selected.append(item)
        else:
            first = selected[0].get("properties", {}).get("datetime")
            current = item.get("properties", {}).get("datetime")
            if first and current:
                from datetime import datetime

                delta = abs(
                    (
                        datetime.fromisoformat(first.replace("Z", "+00:00"))
                        - datetime.fromisoformat(current.replace("Z", "+00:00"))
                    ).days
                )
                if delta >= min_gap_days:
                    selected.append(item)
                    break
    return selected


def default_dates() -> tuple[date, date]:
    end = date.today()
    return end - timedelta(days=365), end
