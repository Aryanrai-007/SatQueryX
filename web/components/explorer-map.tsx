"use client";

import { MapContainer, TileLayer, Marker, useMap, useMapEvents, Polygon } from "react-leaflet";
import L from "leaflet";
import { useEffect, useMemo, useState } from "react";
import "leaflet/dist/leaflet.css";

function Controller({ center, drawing, onCenterChange, onRoi }: { center:[number,number]; drawing:boolean; onCenterChange:(c:[number,number])=>void; onRoi:(r:[number,number][][])=>void }) {
  const map=useMap();
  const [points,setPoints]=useState<[number,number][]>([]);
  useEffect(() => {
    map.flyTo(center, map.getZoom(), { duration: 0.6 });
    return undefined;
  }, [center, map]);
  useEffect(() => {
    if (!drawing) setPoints([]);
    return undefined;
  }, [drawing]);
  useMapEvents({
    click(e){
      const c:[number,number]=[Number(e.latlng.lat.toFixed(5)),Number(e.latlng.lng.toFixed(5))];
      if(drawing) setPoints(p=>[...p,c]); else onCenterChange(c);
    },
    dblclick(){
      if(drawing&&points.length>=3){onRoi([points]);setPoints([]);}
    },
  });
  return <>{points.length>1&&<Polygon positions={points} pathOptions={{color:"#3DD6D0",weight:2,dashArray:"5 5",fillOpacity:0.08}}/>}</>;
}

export default function ExplorerMap({center,onCenterChange,drawing,onRoi,satellite=true}:{center:[number,number];onCenterChange:(c:[number,number])=>void;drawing:boolean;onRoi:(r:[number,number][][])=>void;satellite?:boolean}) {
  const icon=useMemo(()=>new L.Icon({iconUrl:"https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",iconRetinaUrl:"https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",shadowUrl:"https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",iconSize:[25,41],iconAnchor:[12,41]}),[]);
  const tiles=satellite
    ? "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    : "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
  return <div className="satqueryx-explorer-map-shell">
    <MapContainer center={center} zoom={5} doubleClickZoom={false} className="satqueryx-explorer-map" style={{height:"100%",width:"100%",background:"#050b10",zIndex:0}}>
      <TileLayer attribution={satellite ? "© Esri" : "© OpenStreetMap contributors"} url={tiles}/>
      <Marker position={center} icon={icon}/>
      <Controller center={center} drawing={drawing} onCenterChange={onCenterChange} onRoi={onRoi}/>
    </MapContainer>
    {drawing&&<div className="pointer-events-none absolute left-1/2 top-20 z-20 -translate-x-1/2 rounded-lg border border-cyan-300/40 bg-black/85 px-4 py-2 text-[11px] text-cyan-200 backdrop-blur">Click points to define ROI · double-click to finish</div>}
  </div>;
}
