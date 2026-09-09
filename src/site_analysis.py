from __future__ import annotations
from math import floor
from functools import lru_cache
from typing import Any
import numpy as np
import requests
import rasterio
from pyproj import Geod
from rasterio.windows import from_bounds

OVERPASS_URLS=["https://overpass.private.coffee/api/interpreter","https://overpass-api.de/api/interpreter"]
EARTH_SEARCH_URL="https://earth-search.aws.element84.com/v1/search"
OPENTOPO_URL="https://api.opentopodata.org/v1/srtm30m"
WORLDCOVER_PREFIX="https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
APP_UA="SatQueryX/1.0 (remote-sensing-analysis)"
HTTP_TIMEOUT=(3,8); GDAL_TIMEOUT="5"
WORLD_COVER_CLASSES={10:"Tree cover",20:"Shrubland",30:"Grassland",40:"Cropland",50:"Built-up",60:"Bare / sparse vegetation",70:"Snow / ice",80:"Permanent water",90:"Herbaceous wetland",95:"Mangroves",100:"Moss / lichen"}

def _bbox_area_m2(bbox):
    west,south,east,north=bbox; lat=np.deg2rad((south+north)/2); return max(0.0,(east-west)*111320*np.cos(lat)*(north-south)*111320)

def polygon_area_m2(geometry: dict[str,Any]|None)->float|None:
    if not geometry:return None
    geod=Geod(ellps="WGS84"); typ=geometry.get("type"); coords=geometry.get("coordinates",[])
    try:
        if typ=="Polygon":
            outer=coords[0]; area,_=geod.polygon_area_perimeter([p[0] for p in outer],[p[1] for p in outer]); total=abs(area)
            for ring in coords[1:]: total-=abs(geod.polygon_area_perimeter([p[0] for p in ring],[p[1] for p in ring])[0])
            return max(0.0,total)
        if typ=="MultiPolygon": return sum(polygon_area_m2({"type":"Polygon","coordinates":p}) or 0 for p in coords)
    except Exception:return None
    return None

def _remote_array(url,bbox,max_pixels=500_000):
    with rasterio.Env(GDAL_HTTP_USERAGENT=APP_UA,GDAL_HTTP_MULTIRANGE="SERIAL",GDAL_HTTP_MAX_RETRY="0",GDAL_HTTP_TIMEOUT=GDAL_TIMEOUT,GDAL_HTTP_CONNECTTIMEOUT="3",CPL_VSIL_CURL_USE_HEAD="NO",GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
        with rasterio.open(f"/vsicurl/{url}") as ds:
            if ds.transform is None or ds.crs is None: raise ValueError("Remote raster has incomplete georeferencing")
            window=from_bounds(*bbox,transform=ds.transform).round_offsets().round_lengths().intersection(rasterio.windows.Window(0,0,ds.width,ds.height))
            if window.width<=0 or window.height<=0: raise ValueError("Remote raster does not overlap AOI")
            scale=min(1.0,(max_pixels/max(1.0,window.width*window.height))**.5); h=max(1,int(window.height*scale)); w=max(1,int(window.width*scale))
            return ds.read(1,window=window,out_shape=(h,w),masked=True).astype(np.float32)

def _worldcover_urls(bbox):
    west,south,east,north=bbox; urls=[]
    for lat in range(floor(south/3)*3,floor(north/3)*3+1,3):
        for lon in range(floor(west/3)*3,floor(east/3)*3+1,3):
            if len(urls)>=4:return urls
            tile=f"{'N' if lat>=0 else 'S'}{abs(lat):02d}{'E' if lon>=0 else 'W'}{abs(lon):03d}"
            urls.append(f"{WORLDCOVER_PREFIX}/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")
    return urls

def _worldcover_stats(bbox):
    counts={}; total=0
    for url in _worldcover_urls(bbox):
        try: arr=_remote_array(url,bbox)
        except Exception: continue
        vals=arr[np.isfinite(arr)].astype(np.uint8); vals=vals[vals!=0]
        if not vals.size:continue
        u,c=np.unique(vals,return_counts=True)
        for cls,n in zip(u.tolist(),c.tolist()):counts[int(cls)]=counts.get(int(cls),0)+int(n); total+=int(n)
    if not total:return None
    area=_bbox_area_m2(bbox); rows=[]
    for cls,count in sorted(counts.items(),key=lambda x:x[1],reverse=True):
        frac=count/total; rows.append({"class":cls,"label":WORLD_COVER_CLASSES.get(cls,f"Class {cls}"),"pixels":count,"fraction":frac,"area_m2":area*frac,"area_ha":area*frac/10000})
    return {"source":"ESA WorldCover 2021 v200 (10 m)","rows":rows,"pixel_count":total}

def _dem_stats_srtm(bbox):
    west,south,east,north=bbox; points=[(south,west),(south,(west+east)/2),(south,east),((south+north)/2,west),((south+north)/2,(west+east)/2),((south+north)/2,east),(north,west),(north,(west+east)/2),(north,east)]
    loc="|".join(f"{a:.6f},{b:.6f}" for a,b in points); r=requests.get(OPENTOPO_URL,params={"locations":loc,"interpolation":"bilinear"},headers={"User-Agent":APP_UA},timeout=HTTP_TIMEOUT); r.raise_for_status(); vals=[float(x["elevation"]) for x in r.json().get("results",[]) if x.get("elevation") is not None]
    if not vals:return None
    z=np.asarray(vals,dtype=np.float32); return {"source":"SRTM 30 m via Open Topo Data (fallback)","model":"SRTM digital elevation model","mean_m":float(z.mean()),"median_m":float(np.median(z)),"min_m":float(z.min()),"max_m":float(z.max()),"relief_m":float(z.max()-z.min()),"samples":len(z)}

def _dem_stats_copernicus(bbox):
    r=requests.post(EARTH_SEARCH_URL,json={"collections":["cop-dem-glo-30"],"bbox":list(bbox),"limit":2},headers={"User-Agent":APP_UA},timeout=HTTP_TIMEOUT); r.raise_for_status(); features=r.json().get("features",[]); vals=[]
    for item in features[:2]:
        href=next((item.get("assets",{}).get(k,{}).get("href") for k in ("data","dem","elevation","visual") if item.get("assets",{}).get(k,{}).get("href")),None)
        if not href:continue
        try:
            a=_remote_array(href,bbox,250000); f=a[np.isfinite(a)]
            if f.size:vals.append(f.astype(np.float32))
        except Exception:pass
    if not vals:return None
    z=np.concatenate(vals); return {"source":"Copernicus DEM GLO-30 via Earth Search","model":"Copernicus DEM GLO-30 digital surface model (DSM)","mean_m":float(z.mean()),"median_m":float(np.median(z)),"min_m":float(z.min()),"max_m":float(z.max()),"relief_m":float(z.max()-z.min()),"samples":len(z)}

def _dem_stats(bbox):
    try:
        x=_dem_stats_copernicus(bbox)
        if x:return x
    except Exception:pass
    try:return _dem_stats_srtm(bbox)
    except Exception:return None

def _osm_waterways(bbox):
    west,south,east,north=bbox; pad=.005; qwest,qsouth,qeast,qnorth=west-pad,south-pad,east+pad,north+pad
    query=f'''[out:json][timeout:6];(way[waterway~"^(river|stream|canal|drain)$"]({qsouth},{qwest},{qnorth},{qeast});relation[waterway~"^(river|stream|canal)$"]({qsouth},{qwest},{qnorth},{qeast});way[natural=water]({qsouth},{qwest},{qnorth},{qeast});relation[natural=water]({qsouth},{qwest},{qnorth},{qeast}););out tags;'''
    last=None
    for endpoint in OVERPASS_URLS:
        try:
            r=requests.post(endpoint,data=query.encode(),headers={"User-Agent":APP_UA},timeout=HTTP_TIMEOUT);r.raise_for_status(); els=r.json().get("elements",[]); names=[];types=[]
            for e in els:
                t=e.get("tags",{}); n=t.get("name") or t.get("name:en"); k=t.get("waterway") or t.get("natural")
                if n and n not in names:names.append(n)
                if k and k not in types:types.append(k)
            return {"names":names[:20],"types":types[:20],"count":len(els),"source":f"OpenStreetMap / Overpass API ({endpoint.split('/')[2]})","context_radius_degrees":pad}
        except Exception as e:last=e
    raise RuntimeError(f"All OpenStreetMap Overpass endpoints failed: {last}")

def building_footprints(bbox):
    west,south,east,north=bbox; query=f'''[out:json][timeout:6];way[building]({south},{west},{north},{east});relation[building]({south},{west},{north},{east});out tags;'''
    last=None
    for endpoint in OVERPASS_URLS:
        try:
            r=requests.post(endpoint,data=query.encode(),headers={"User-Agent":APP_UA},timeout=HTTP_TIMEOUT);r.raise_for_status(); els=r.json().get("elements",[]); return {"count":len(els),"source":f"OpenStreetMap / Overpass API ({endpoint.split('/')[2]})","scope":"mapped building footprints inside AOI bbox"}
        except Exception as e:last=e
    raise RuntimeError(f"Building footprint lookup failed: {last}")

@lru_cache(maxsize=32)
def _cached_site_sources(bbox):
    lim=[]
    try:wc=_worldcover_stats(bbox)
    except Exception as e:wc=None;lim.append(f"ESA WorldCover unavailable: {e}")
    if wc is None:lim.append("ESA WorldCover returned no usable pixels for the AOI.")
    try:elev=_dem_stats(bbox)
    except Exception as e:elev=None;lim.append(f"Elevation sources unavailable: {e}")
    if elev is None:lim.append("Copernicus DEM GLO-30 and SRTM fallback returned no usable elevation values.")
    try:water=_osm_waterways(bbox)
    except Exception as e:water=None;lim.append(f"OpenStreetMap water-feature lookup unavailable: {e}")
    try:buildings=building_footprints(bbox)
    except Exception as e:buildings=None;lim.append(f"OpenStreetMap building lookup unavailable: {e}")
    return wc,elev,water,buildings,tuple(lim)

def build_site_summary(bbox,ndvi=None,geometry=None):
    key=tuple(round(float(v),6) for v in bbox);wc,elev,water,buildings,limitations=_cached_site_sources(key); exact=polygon_area_m2(geometry); area_m2=exact if exact is not None else _bbox_area_m2(key)
    summary={"area_ha":area_m2/10000,"area_basis":"exact drawn polygon" if exact is not None else "bounding-box footprint","worldcover":wc,"elevation":elev,"waterways":water,"buildings":buildings,"limitations":list(limitations)}
    if ndvi:summary["ndvi_vegetated_area_ha"]=summary["area_ha"]*float(ndvi.get("vegetated_fraction",0))
    return summary
