from __future__ import annotations

from math import floor
from typing import Any

import numpy as np
import requests
import rasterio
from rasterio.windows import from_bounds

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
WORLDCOVER_PREFIX = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
APP_UA = "SatQueryX/1.0 (remote-sensing-analysis; contact=project)"

WORLD_COVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow / ice",
    80: "Permanent water",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss / lichen",
}


def _bbox_area_m2(bbox: tuple[float, float, float, float]) -> float:
    west, south, east, north = bbox
    lat = np.deg2rad((south + north) / 2.0)
    dy = (north - south) * 111_320.0
    dx = (east - west) * 111_320.0 * float(np.cos(lat))
    return max(0.0, dx * dy)


def _remote_array(url: str, bbox: tuple[float, float, float, float], max_pixels: int = 2_000_000) -> np.ndarray:
    with rasterio.Env(
        GDAL_HTTP_USERAGENT=APP_UA,
        GDAL_HTTP_MULTIRANGE="SERIAL",
        GDAL_HTTP_MAX_RETRY="2",
        GDAL_HTTP_RETRY_CODES="429,500,502,503,504",
        CPL_VSIL_CURL_USE_HEAD="NO",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    ):
        with rasterio.open(f"/vsicurl/{url}") as ds:
            window = from_bounds(*bbox, transform=ds.transform)
            window = window.round_offsets().round_lengths()
            window = window.intersection(rasterio.windows.Window(0, 0, ds.width, ds.height))
            if window.width <= 0 or window.height <= 0:
                raise ValueError("Remote raster does not overlap AOI")
            scale = min(1.0, (max_pixels / max(1.0, window.width * window.height)) ** 0.5)
            out_h = max(1, int(window.height * scale))
            out_w = max(1, int(window.width * scale))
            return ds.read(1, window=window, out_shape=(out_h, out_w), masked=True).astype(np.float32)


def _worldcover_urls(bbox: tuple[float, float, float, float]) -> list[str]:
    west, south, east, north = bbox
    urls = []
    for lat in range(floor(south), floor(north) + 1):
        for lon in range(floor(west), floor(east) + 1):
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
    rows = []
    for cls, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        label = WORLD_COVER_CLASSES.get(cls, f"Class {cls}")
        frac = count / total
        rows.append({"class": int(cls), "label": label, "pixels": count, "fraction": frac})
    area_m2 = _bbox_area_m2(bbox)
    for row in rows:
        row["area_m2"] = area_m2 * row["fraction"]
        row["area_ha"] = row["area_m2"] / 10_000.0
    return {"source": "ESA WorldCover 2021 v200", "rows": rows, "pixel_count": total}


def _dem_stats(bbox: tuple[float, float, float, float]) -> dict[str, Any] | None:
    payload = {"collections": ["cop-dem-glo-30"], "bbox": list(bbox), "limit": 10}
    response = requests.post(EARTH_SEARCH_URL, json=payload, headers={"User-Agent": APP_UA}, timeout=20)
    response.raise_for_status()
    features = response.json().get("features", [])
    if not features:
        return None
    values = []
    for item in features:
        href = None
        for key, asset in item.get("assets", {}).items():
            if key in {"data", "dem", "elevation"} or "tif" in str(asset.get("type", "")):
                href = asset.get("href")
                if href:
                    break
        if not href:
            continue
        try:
            arr = _remote_array(href, bbox)
            vals = arr[np.isfinite(arr)]
            vals = vals[vals > -1000]
            if vals.size:
                values.append(vals)
        except Exception:
            continue
    if not values:
        return None
    z = np.concatenate(values)
    return {
        "source": "Copernicus DEM GLO-30",
        "model": "Digital Surface Model (DSM)",
        "mean_m": float(np.mean(z)),
        "median_m": float(np.median(z)),
        "min_m": float(np.min(z)),
        "max_m": float(np.max(z)),
        "relief_m": float(np.max(z) - np.min(z)),
        "samples": int(z.size),
    }


def _osm_waterways(bbox: tuple[float, float, float, float]) -> dict[str, Any]:
    west, south, east, north = bbox
    query = f'''
    [out:json][timeout:15];
    (
      way[waterway~"^(river|stream|canal|drain)$"]({south},{west},{north},{east});
      relation[waterway~"^(river|stream|canal)$"]({south},{west},{north},{east});
      way[natural=water]({south},{west},{north},{east});
      relation[natural=water]({south},{west},{north},{east});
    );
    out tags center;
    '''
    response = requests.post(OVERPASS_URL, data=query.encode("utf-8"), headers={"User-Agent": APP_UA}, timeout=20)
    response.raise_for_status()
    elements = response.json().get("elements", [])
    names = []
    types = []
    for element in elements:
        tags = element.get("tags", {})
        name = tags.get("name") or tags.get("name:en")
        kind = tags.get("waterway") or tags.get("natural")
        if name and name not in names:
            names.append(name)
        if kind and kind not in types:
            types.append(kind)
    return {"names": names[:20], "types": types[:20], "count": len(elements), "source": "OpenStreetMap / Overpass API"}


def build_site_summary(bbox: tuple[float, float, float, float], ndvi: dict[str, Any] | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "bbox_area_ha": _bbox_area_m2(bbox) / 10_000.0,
        "worldcover": None,
        "elevation": None,
        "waterways": None,
        "limitations": [],
    }
    try:
        summary["worldcover"] = _worldcover_stats(bbox)
    except Exception as exc:
        summary["limitations"].append(f"ESA WorldCover unavailable: {exc}")
    try:
        summary["elevation"] = _dem_stats(bbox)
    except Exception as exc:
        summary["limitations"].append(f"Copernicus DEM unavailable: {exc}")
    try:
        summary["waterways"] = _osm_waterways(bbox)
    except Exception as exc:
        summary["limitations"].append(f"OpenStreetMap water-feature lookup unavailable: {exc}")
    if ndvi:
        summary["ndvi_vegetated_area_ha"] = summary["bbox_area_ha"] * float(ndvi.get("vegetated_fraction", 0.0))
    return summary
