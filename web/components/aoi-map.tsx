"use client";

import { useMemo } from "react";
import { MapContainer, TileLayer, Rectangle, CircleMarker, useMapEvents } from "react-leaflet";
import "leaflet/dist/leaflet.css";

type Props = {
  center: [number, number];
  onCenterChange: (center: [number, number]) => void;
};

function MapInteraction({ onCenterChange }: { onCenterChange: Props["onCenterChange"] }) {
  useMapEvents({
    click(e) {
      onCenterChange([Number(e.latlng.lat.toFixed(5)), Number(e.latlng.lng.toFixed(5))]);
    },
  });
  return null;
}

export default function AOIMap({ center, onCenterChange }: Props) {
  const bounds = useMemo<[number, number][]>(() => [
    [center[0] - 0.018, center[1] - 0.024],
    [center[0] + 0.018, center[1] + 0.024],
  ], [center]);

  return (
    <div className="relative h-[310px] overflow-hidden rounded-2xl border border-white/10 bg-slate-950">
      <MapContainer center={center} zoom={11} scrollWheelZoom className="h-full w-full">
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <MapInteraction onCenterChange={onCenterChange} />
        <Rectangle bounds={bounds} pathOptions={{ color: "#22d3ee", weight: 2, fillOpacity: 0.08 }} />
        <CircleMarker center={center} radius={5} pathOptions={{ color: "#67e8f9", fillColor: "#22d3ee", fillOpacity: 1 }} />
      </MapContainer>
      <div className="pointer-events-none absolute left-3 top-3 z-[500] rounded-lg border border-cyan-300/20 bg-slate-950/80 px-3 py-2 text-[10px] uppercase tracking-[0.18em] text-cyan-200 backdrop-blur">
        AOI interaction · click map to reposition
      </div>
    </div>
  );
}
