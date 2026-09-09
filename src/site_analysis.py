from __future__ import annotations

from math import floor
from functools import lru_cache
from typing import Any

import numpy as np
import requests
import rasterio
from rasterio.windows import from_bounds

OVERPASS_URLS = [
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]
EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"
OPENTOPO_URL = "https://api.opentopodata.org/v1/srtm30m"
WORLDCOVER_PREFIX = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
APP_UA = "SatQueryX/1.0 (remote-sensing-analysis)"

HTTP_TIMEOUT = (3, 8)
GDAL_TIMEOUT = "5"

WORLD_COVER_CLASSES = {
    10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
    50: "Built-up", 60: "Bare / sparse vegetation", 70: "Snow / ice",
    80: "Permanent water", 90: "Herbaceous wetland", 95: "Mangroves", 100: "Moss / lichen",
}


def _bbox_area_m2(bbox: tuple[float, float, float, float]) -> float:
    west, south, east, north = bbox
    lat = np.deg2rad((south + north) / 2.0)
    dy = (north - south) * 111_320.0
    dx = (east - west) * 111_320.0 * float(np.cos(lat))
    return max(0.0, dx * dy)


def _remote_array(url: str, bbox: tuple[float, float, float, float], max_pixels: int = 500_000) -> np.ndarray:
    with rasterio.Env(
        GDAL_HTTP_USERAGENT=APP_UA,
        GDAL_HTTP_MULTIRANGE="SERIAL",
        GDAL_HTTP_MAX_RETRY="0",
        GDAL_HTTP_TIMEOUT=GDAL_TIMEOUT,
        GDAL_HTTP_CONNECTTIMEOUT="3",
        CPL_VSIL_CURL_USE_HEAD="NO",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    ):
        with rasterio.open(f"/vsicurl/{url}") as ds:
            if ds.transform is None or ds.crs is None:
                raise ValueError("Remote raster has incomplete georeferencing")
            window = from_bounds(*bbox, transform=ds.transform).round_offsets().round_lengths()
            window = window.intersection(rasterio.windows.Window(0, 0, ds.width, ds.height))
            if window.width <= 0 or window.height <= 0:
                raise ValueError("Remote raster does not overlap AOI")
            scale = min(1.0, (max_pixels / max(1.0, window.width * window.height)) ** 0.5)
            out_h = max(1, int(window.height * scale))
            out_w = max(1, int(window.width * scale))
            return ds.read(1, window=window, out_shape=(out_h, out_w), masked=True).astype(np.float32)


def _worldcover_urls(bbox: tuple[float, float, float, float]) -> list[str]:
    """Return the ESA WorldCover 3° x 3° tiles intersecting the AOI.

    WorldCover v200 map tiles are 3° x 3°, not 1° x 1°. The old 1-degree
    construction silently produced nonexistent filenames and therefore no
    land-cover results.
    """
    west, south, east, north = bbox
    min_lat = floor(south / 3.0) * 3
    max_lat = floor(north / 3.0) * 3
    min_lon = floor(west / 3.0) * 3
    max_lon = floor(east / 3.0) * 3
    urls: list[str] = []
    for lat in range(min_lat, max_lat + 1, 3):
        for lon in range(min_lon, max_lon + 1, 3):
            if len(urls) >= 4:
                return urls
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            tile = f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}"
            urls.append(f"{WORLDCOVER_PREFIX}/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")
    return urls


def _worldcover_stats(bbox: tuple[float, float, float, float]) -> dict[str, Any] | None:
    counts: dict[int, int] = {}
    total = 0
    for url in _worldcover_urls(bbox):
        try:
            arr = _remote_array(url, bbox)
        except Exception:
            continue
        vals = arr[np.isfinite(arr)].astype(np.uint8)
        vals = vals[vals != 0]
        if vals.size == 0:
            continue
        uniq, cnt = np.unique(vals, return_counts=True)
        for cls, n in zip(uniq.tolist(), cnt.tolist()):
            counts[int(cls)] = counts.get(int(cls), 0) + int(n)
            total += int(n)
    if total == 0:
        return None
    area_m2 = _bbox_area_m2(bbox)
    rows = []
    for cls, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        frac = count / total
        rows.append({
            "class": int(cls),
            "label": WORLD_COVER_CLASSES.get(cls, f"Class {cls}"),
            "pixels": count,
            "fraction": frac,
            "area_m2": area_m2 * frac,
            "area_ha": area_m2 * frac / 10_000.0,
        })
    return {"source": "ESA WorldCover 2021 v200 (10 m)", "rows": rows, "pixel_count": total}


def _dem_stats_srtm(bbox: tuple[float, float, float, float]) -> dict[str, Any] | None:
    west, south, east, north = bbox
    points = [
        (south, west), (south, (west + east) / 2), (south, east),
        ((south + north) / 2, west), ((south + north) / 2, (west + east) / 2), ((south + north) / 2, east),
        (north, west), (north, (west + east) / 2), (north, east),
    ]
    locations = "|".join(f"{lat:.6f},{lon:.6f}" for lat, lon in points)
    response = requests.get(OPENTOPO_URL, params={"locations": locations, "interpolation": "bilinear"}, headers={"User-Agent": APP_UA}, timeout=HTTP_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    values = [float(item["elevation"]) for item in payload.get("results", []) if item.get("elevation") is not None]
    if not values:
        return None
    z = np.asarray(values, dtype=np.float32)
    return {
        "source": "SRTM 30 m via Open Topo Data (fallback)",
        "model": "SRTM digital elevation model",
        "mean_m": float(np.mean(z)), "median_m": float(np.median(z)),
        "min_m": float(np.min(z)), "max_m": float(np.max(z)),
        "relief_m": float(np.max(z) - np.min(z)), "samples": int(z.size),
    }


def _dem_stats_copernicus(bbox: tuple[float, float, float, float]) -> dict[str, Any] | None:
    """Read the actual Copernicus DEM GLO-30 COG through Earth Search STAC."""
    response = requests.post(
        EARTH_SEARCH_URL,
        json={"collections": ["cop-dem-glo-30"], "bbox": list(bbox), "limit": 2},
        headers={"User-Agent": APP_UA},
        timeout=HTTP_TIMEOUT,
    )
    response.raise_for_status()
    features = response.json().get("features", [])
    if not features:
        return None
    values: list[np.ndarray] = []
    for item in features[:2]:
        assets = item.get("assets", {})
        href = None
        for key in ("data", "dem", "elevation", "visual"):
            candidate = assets.get(key, {}).get("href")
            if candidate:
                href = candidate
                break
        if not href:
            continue
        try:
            arr = _remote_array(href, bbox, max_pixels=250_000)
            finite = arr[np.isfinite(arr)]
            if finite.size:
                values.append(finite.astype(np.float32))
        except Exception:
            continue
    if not values:
        return None
    z = np.concatenate(values)
    return {
        "source": "Copernicus DEM GLO-30 via Earth Search",
        "model": "Copernicus DEM GLO-30 digital surface model (DSM)",
        "mean_m": float(np.mean(z)), "median_m": float(np.median(z)),
        "min_m": float(np.min(z)), "max_m": float(np.max(z)),
        "relief_m": float(np.max(z) - np.min(z)), "samples": int(z.size),
    }


def _dem_stats(bbox: tuple[float, float, float, float]) -> dict[str, Any] | None:
    try:
        result = _dem_stats_copernicus(bbox)
        if result:
            return result
    except Exception:
        pass
    return _dem_stats_srtm(bbox)


def _osm_waterways(bbox: tuple[float, float, float, float]) -> dict[str, Any]:
    west, south, east, north = bbox
    # Search a small context ring around the AOI so nearby rivers/canals are useful,
    # while explicitly reporting that these are mapped features in the context bbox.
    pad = 0.005
    qwest, qsouth, qeast, qnorth = west - pad, south - pad, east + pad, north + pad
    query = f'''[out:json][timeout:6];(
      way[waterway~"^(river|stream|canal|drain)$"]({qsouth},{qwest},{qnorth},{qeast});
      relation[waterway~"^(river|stream|canal)$"]({qsouth},{qwest},{qnorth},{qeast});
      way[natural=water]({qsouth},{qwest},{qnorth},{qeast});
      relation[natural=water]({qsouth},{qwest},{qnorth},{qeast});
    );out tags;'''
    last_error = None
    for endpoint in OVERPASS_URLS:
        try:
            response = requests.post(endpoint, data=query.encode("utf-8"), headers={"User-Agent": APP_UA}, timeout=HTTP_TIMEOUT)
            response.raise_for_status()
            elements = response.json().get("elements", [])
            names, types = [], []
            for element in elements:
                tags = element.get("tags", {})
                name = tags.get("name") or tags.get("name:en")
                kind = tags.get("waterway") or tags.get("natural")
                if name and name not in names:
                    names.append(name)
                if kind and kind not in types:
                    types.append(kind)
            return {
                "names": names[:20], "types": types[:20], "count": len(elements),
                "source": f"OpenStreetMap / Overpass API ({endpoint.split('/')[2]})",
                "context_radius_degrees": pad,
            }
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"All OpenStreetMap Overpass endpoints failed: {last_error}")


@lru_cache(maxsize=32)
def _cached_site_sources(bbox: tuple[float, float, float, float]) -> tuple[Any, Any, Any, tuple[str, ...]]:
    limitations: list[str] = []
    try:
        wc = _worldcover_stats(bbox)
        if wc is None:
            limitations.append("ESA WorldCover returned no usable pixels for the AOI.")
    except Exception as exc:
        wc = None
        limitations.append(f"ESA WorldCover unavailable: {exc}")
    try:
        elev = _dem_stats(bbox)
        if elev is None:
            limitations.append("Copernicus DEM GLO-30 and SRTM fallback returned no usable elevation values.")
    except Exception as exc:
        elev = None
        limitations.append(f"Elevation sources unavailable: {exc}")
    try:
        water = _osm_waterways(bbox)
    except Exception as exc:
        water = None
        limitations.append(f"OpenStreetMap water-feature lookup unavailable: {exc}")
    return wc, elev, water, tuple(limitations)


def build_site_summary(bbox: tuple[float, float, float, float], ndvi: dict[str, Any] | None = None) -> dict[str, Any]:
    key = tuple(round(float(v), 6) for v in bbox)
    wc, elev, water, limitations = _cached_site_sources(key)
    summary: dict[str, Any] = {
        "bbox_area_ha": _bbox_area_m2(key) / 10_000.0,
        "worldcover": wc,
        "elevation": elev,
        "waterways": water,
        "limitations": list(limitations),
    }
    if ndvi:
        summary["ndvi_vegetated_area_ha"] = summary["bbox_area_ha"] * float(ndvi.get("vegetated_fraction", 0.0))
    return summary
