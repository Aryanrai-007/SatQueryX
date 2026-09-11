"use client";

import { useEffect, useMemo, useState } from "react";
import { MapContainer, TileLayer, Rectangle, Polygon, CircleMarker, useMap, useMapEvents } from "react-leaflet";
import type { LatLng, LatLngBoundsExpression } from "leaflet";
import "leaflet/dist/leaflet.css";

export type MapMode = "pan" | "rectangle" | "polygon";
export type Snippet = {
  type: "rectangle" | "polygon";
  coordinates: [number, number][];
};

type Props = {
  center: [number, number];
  onCenterChange: (center: [number, number]) => void;
  onSnippetChange: (snippet: Snippet | null) => void;
};

function DrawingController({ mode, onCenterChange, onSnippetChange }: { mode: MapMode; onCenterChange: Props["onCenterChange"]; onSnippetChange: Props["onSnippetChange"] }) {
  const map = useMap();
  const [start, setStart] = useState<LatLng | null>(null);
  const [draftBounds, setDraftBounds] = useState<LatLngBoundsExpression | null>(null);
  const [polygon, setPolygon] = useState<[number, number][]>([]);

  useEffect(() => {
    setStart(null);
    setDraftBounds(null);
    setPolygon([]);
    if (mode === "polygon") map.doubleClickZoom.disable();
    else map.doubleClickZoom.enable();
    if (mode === "pan") map.dragging.enable();
    return () => {
      map.dragging.enable();
      map.doubleClickZoom.enable();
    };
  }, [mode, map]);

  useMapEvents({
    click(event) {
      onCenterChange([Number(event.latlng.lat.toFixed(5)), Number(event.latlng.lng.toFixed(5))]);
      if (mode === "polygon") {
        setPolygon((points) => [...points, [Number(event.latlng.lat.toFixed(6)), Number(event.latlng.lng.toFixed(6))]]);
      }
    },
    dblclick(event) {
      if (mode !== "polygon") return;
      event.originalEvent.preventDefault();
      setPolygon((points) => {
        const next = points.length >= 3 ? points : [...points, [Number(event.latlng.lat.toFixed(6)), Number(event.latlng.lng.toFixed(6))]];
        if (next.length >= 3) onSnippetChange({ type: "polygon", coordinates: next });
        return next;
      });
    },
    mousedown(event) {
      if (mode !== "rectangle") return;
      map.dragging.disable();
      setStart(event.latlng);
      setDraftBounds([[event.latlng.lat, event.latlng.lng], [event.latlng.lat, event.latlng.lng]]);
    },
    mousemove(event) {
      if (mode !== "rectangle" || !start) return;
      setDraftBounds([[start.lat, start.lng], [event.latlng.lat, event.latlng.lng]]);
    },
    mouseup(event) {
      if (mode !== "rectangle" || !start) return;
      const south = Math.min(start.lat, event.latlng.lat);
      const north = Math.max(start.lat, event.latlng.lat);
      const west = Math.min(start.lng, event.latlng.lng);
      const east = Math.max(start.lng, event.latlng.lng);
      if (Math.abs(north - south) > 0.001 && Math.abs(east - west) > 0.001) {
        onSnippetChange({ type: "rectangle", coordinates: [[south, west], [north, east]] });
      }
      setStart(null);
      setDraftBounds(null);
      map.dragging.enable();
    },
  });

  return (
    <>
      {draftBounds && <Rectangle bounds={draftBounds} pathOptions={{ color: "#22d3ee", weight: 2, dashArray: "6 5", fillOpacity: 0.12 }} />}
      {polygon.length >= 2 && <Polygon positions={polygon} pathOptions={{ color: "#a78bfa", weight: 2, dashArray: "6 5", fillOpacity: 0.12 }} />}
    </>
  );
}

function MapViewReset({ center }: { center: [number, number] }) {
  const map = useMap();
  useEffect(() => {
    map.setView(center, map.getZoom(), { animate: true });
  }, [center, map]);
  return null;
}

export default function AOIMap({ center, onCenterChange, onSnippetChange }: Props) {
  const [mode, setMode] = useState<MapMode>("pan");
  const [snippet, setSnippet] = useState<Snippet | null>(null);
  const bounds = useMemo<[number, number][]>(() => [
    [center[0] - 0.018, center[1] - 0.024],
    [center[0] + 0.018, center[1] + 0.024],
  ], [center]);

  function updateSnippet(next: Snippet | null) {
    setSnippet(next);
    onSnippetChange(next);
  }

  function clearSnippet() {
    updateSnippet(null);
    setMode("pan");
  }

  return (
    <div className="relative h-[520px] overflow-hidden rounded-[24px] border border-cyan-300/15 bg-slate-950 shadow-[0_30px_90px_rgba(0,0,0,.35)] lg:h-[620px]">
      <MapContainer center={center} zoom={11} scrollWheelZoom className="h-full w-full">
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <MapViewReset center={center} />
        <DrawingController mode={mode} onCenterChange={onCenterChange} onSnippetChange={updateSnippet} />
        {mode === "pan" && <Rectangle bounds={bounds} pathOptions={{ color: "#22d3ee", weight: 2, fillOpacity: 0.035, dashArray: "7 7" }} />}
        <CircleMarker center={center} radius={6} pathOptions={{ color: "#e0f2fe", weight: 2, fillColor: "#22d3ee", fillOpacity: 1 }} />
      </MapContainer>

      <div className="absolute left-4 top-4 z-[500] max-w-[330px] rounded-2xl border border-white/10 bg-slate-950/90 px-4 py-3 shadow-2xl backdrop-blur-xl">
        <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.2em] text-cyan-300"><span className="h-2 w-2 rounded-full bg-cyan-300 shadow-[0_0_14px_rgba(34,211,238,.9)]"/>Mission AOI</div>
        <div className="mt-1 text-xs text-slate-300">{mode === "rectangle" ? "Drag to trace a rectangular image snippet." : mode === "polygon" ? "Click vertices, then double-click to close the snippet." : "Pan the map or choose a trace mode below."}</div>
      </div>

      <div className="absolute bottom-4 left-4 right-4 z-[500] flex flex-col gap-3 rounded-2xl border border-white/10 bg-slate-950/92 p-3 shadow-2xl backdrop-blur-xl sm:flex-row sm:items-center sm:justify-between">
        <div className="flex flex-wrap gap-2">
          <button onClick={() => setMode("pan")} className={`rounded-xl border px-3 py-2 text-[10px] font-semibold uppercase tracking-wider transition ${mode === "pan" ? "border-cyan-300/35 bg-cyan-300/10 text-cyan-100" : "border-white/10 text-slate-400 hover:border-white/20"}`}>Pan</button>
          <button onClick={() => setMode("rectangle")} className={`rounded-xl border px-3 py-2 text-[10px] font-semibold uppercase tracking-wider transition ${mode === "rectangle" ? "border-cyan-300/35 bg-cyan-300/10 text-cyan-100" : "border-white/10 text-slate-400 hover:border-white/20"}`}>Trace snippet</button>
          <button onClick={() => setMode("polygon")} className={`rounded-xl border px-3 py-2 text-[10px] font-semibold uppercase tracking-wider transition ${mode === "polygon" ? "border-violet-300/35 bg-violet-300/10 text-violet-100" : "border-white/10 text-slate-400 hover:border-white/20"}`}>Trace polygon</button>
          {snippet && <button onClick={clearSnippet} className="rounded-xl border border-red-300/20 px-3 py-2 text-[10px] font-semibold uppercase tracking-wider text-red-200 hover:bg-red-300/10">Clear</button>}
        </div>
        <div className="text-right text-[10px] text-slate-500">
          {snippet ? <><span className="text-cyan-200">Snippet captured</span><br/>{snippet.type} · {snippet.coordinates.length} vertices</> : <>No snippet selected<br/>Click / trace to define AOI</>}
        </div>
      </div>
    </div>
  );
}
