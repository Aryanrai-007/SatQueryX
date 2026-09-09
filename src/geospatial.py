from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import BinaryIO

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.warp import reproject


@dataclass
class RasterInfo:
    name: str
    width: int
    height: int
    count: int
    dtype: str
    crs: str | None
    resolution_x: float
    resolution_y: float
    bounds: tuple[float, float, float, float]
    transform: object
    descriptions: tuple[str | None, ...]


def open_raster(data: bytes):
    """Open uploaded bytes as an in-memory rasterio dataset."""
    memfile = MemoryFile(data)
    ds = memfile.open()
    return memfile, ds


def raster_info(ds, name: str) -> RasterInfo:
    return RasterInfo(
        name=name,
        width=ds.width,
        height=ds.height,
        count=ds.count,
        dtype=str(ds.dtypes[0]),
        crs=ds.crs.to_string() if ds.crs else None,
        resolution_x=abs(float(ds.transform.a)),
        resolution_y=abs(float(ds.transform.e)),
        bounds=(ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top),
        transform=ds.transform,
        descriptions=tuple(ds.descriptions),
    )


def validate_geospatial(ds) -> list[str]:
    errors: list[str] = []
    if ds.width <= 0 or ds.height <= 0:
        errors.append("Raster has invalid dimensions.")
    if ds.count < 1:
        errors.append("Raster contains no bands.")
    if ds.crs is None:
        errors.append("Raster has no CRS/georeferencing. Supply a georeferenced GeoTIFF for map-aware analysis.")
    if not np.isfinite(ds.transform.a) or not np.isfinite(ds.transform.e):
        errors.append("Raster has invalid pixel resolution.")
    return errors


def read_preview(ds, max_size: int = 1200) -> tuple[np.ndarray, dict]:
    scale = min(1.0, max_size / max(ds.width, ds.height))
    out_w = max(1, int(ds.width * scale))
    out_h = max(1, int(ds.height * scale))
    arr = ds.read(
        out_shape=(min(ds.count, 4), out_h, out_w),
        resampling=Resampling.bilinear,
        masked=True,
    )
    arr = np.ma.filled(arr, np.nan).astype(np.float32)
    return arr, {"width": out_w, "height": out_h}


def normalize_band(band: np.ndarray, low: float | None = None, high: float | None = None) -> np.ndarray:
    valid = np.isfinite(band)
    if not valid.any():
        raise ValueError("Band contains no finite pixels.")
    if low is None:
        low = float(np.nanpercentile(band, 2))
    if high is None:
        high = float(np.nanpercentile(band, 98))
    if high <= low:
        raise ValueError("Band has no usable dynamic range.")
    return np.clip((band - low) / (high - low), 0, 1)


def rgb_preview(ds, rgb_bands: tuple[int, int, int] | None = None) -> np.ndarray:
    if rgb_bands is None:
        if ds.count >= 3:
            rgb_bands = (1, 2, 3)
        else:
            gray, _ = read_preview(ds)
            g = normalize_band(gray[0])
            return np.dstack([g, g, g])
    bands = [ds.read(i, out_shape=(min(1200, ds.height), min(1200, ds.width)), resampling=Resampling.bilinear).astype(np.float32) for i in rgb_bands]
    return np.dstack([normalize_band(b) for b in bands])


def compute_ndvi(ds, red_band: int, nir_band: int) -> np.ndarray:
    if not (1 <= red_band <= ds.count and 1 <= nir_band <= ds.count):
        raise ValueError(f"Band indexes must be between 1 and {ds.count}.")
    red = ds.read(red_band).astype(np.float32)
    nir = ds.read(nir_band).astype(np.float32)
    denom = nir + red
    ndvi = np.divide(nir - red, denom, out=np.full_like(red, np.nan), where=np.abs(denom) > 1e-8)
    return np.clip(ndvi, -1, 1)


def sar_to_db(power: np.ndarray) -> np.ndarray:
    power = np.asarray(power, dtype=np.float32)
    if np.nanmin(power) < 0:
        raise ValueError("SAR power values cannot be negative for linear-to-dB conversion.")
    return 10.0 * np.log10(np.maximum(power, 1e-10))


def reproject_to_reference(src_ds, ref_ds, band: int = 1) -> np.ndarray:
    """Reproject a source band onto the exact reference raster grid."""
    destination = np.full((ref_ds.height, ref_ds.width), np.nan, dtype=np.float32)
    reproject(
        source=src_ds.read(band).astype(np.float32),
        destination=destination,
        src_transform=src_ds.transform,
        src_crs=src_ds.crs,
        dst_transform=ref_ds.transform,
        dst_crs=ref_ds.crs,
        resampling=Resampling.bilinear,
        src_nodata=src_ds.nodata,
        dst_nodata=np.nan,
    )
    return destination


def tile_array(arr: np.ndarray, tile_size: int = 512, overlap: int = 32):
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("tile_size must be positive and overlap must be in [0, tile_size).")
    step = tile_size - overlap
    h, w = arr.shape[-2:]
    for y in range(0, max(1, h - overlap), step):
        for x in range(0, max(1, w - overlap), step):
            y2, x2 = min(y + tile_size, h), min(x + tile_size, w)
            yield (y, y2, x, x2), arr[..., y:y2, x:x2]
            if y2 == h and x2 == w:
                continue


def image_bytes_to_array(data: bytes) -> np.ndarray:
    with Image.open(BytesIO(data)) as im:
        return np.asarray(im.convert("RGB"))


def raster_or_image(data: bytes, name: str):
    """Return (kind, object). GeoTIFF uses an in-memory rasterio dataset; normal images use ndarray."""
    lower = name.lower()
    if lower.endswith((".tif", ".tiff")):
        mem, ds = open_raster(data)
        return "raster", (mem, ds)
    return "image", image_bytes_to_array(data)
