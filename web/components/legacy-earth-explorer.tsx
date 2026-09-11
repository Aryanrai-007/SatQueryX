"use client";

import dynamic from "next/dynamic";
import { useMemo, useState } from "react";
import { Globe2, Search, X, Crosshair, CalendarDays, Layers3 } from "lucide-react";

const ExplorerMap = dynamic(() => import("./explorer-map"), { ssr: false });

const SECTORS = [
  ["Delhi NCR", 28.6139, 77.2090, "Urban density & Yamuna corridor"],
  ["Mumbai Coastline", 19.0760, 72.8777, "Coastal / port monitoring"],
  ["Kolkata Delta", 22.5726, 88.3639, "Urban expansion & delta"],
  ["Gangotri Glacier", 30.93, 79.08, "Glacial retreat"],
  ["Sundarbans", 21.9497, 89.1833, "Wetland monitoring"],
  ["Amazon Front", -3.4653, -58.38, "Vegetation loss"],
] as const;

export function LegacyEarthExplorer({ open, onClose, onSelect }: { open: boolean; onClose: () => void; onSelect: (lat:number, lon:number) => void }) {
  const [center, setCenter] = useState<[number, number]>([28.6139, 77.209]);
  const [query, setQuery] = useState("");
  const [year, setYear] = useState(2024);
  const [satellite, setSatellite] = useState(true);
  const [drawing, setDrawing] = useState(false);
  const [roi, setRoi] = useState<[number, number][][]>([]);
  const filtered = useMemo(() => SECTORS.filter(s => s[0].toLowerCase().includes(query.toLowerCase())), [query]);
  if (!open) return null;
  return <div className="fixed inset-0 z-[9999] isolate flex flex-col overflow-hidden bg-[#05080d]">
    <header className="relative z-30 flex shrink-0 items-center justify-between border-b border-cyan-300/15 bg-[#071019] px-4 py-3 shadow-2xl">
      <div className="flex items-center gap-3"><div className="rounded-lg border border-cyan-300/20 bg-cyan-300/10 p-2"><Globe2 size={18} className="text-cyan-300"/></div><div><div className="text-sm font-semibold tracking-[.08em] text-white">GOD&apos;S EYE · SATQUERYX EARTH</div><div className="text-[9px] uppercase tracking-[.2em] text-slate-600">Global spatial intelligence explorer</div></div></div>
      <div className="flex items-center gap-2"><span className="hidden text-[9px] uppercase tracking-widest text-cyan-300/70 md:block">LIVE SPATIAL WORKSPACE</span><button className="btn btn--ghost btn--sm" onClick={onClose}><X size={15}/></button></div>
    </header>
    <div className="relative z-0 grid min-h-0 flex-1 lg:grid-cols-[1fr_310px]">
      <div className="relative min-h-0 overflow-hidden"><ExplorerMap center={center} onCenterChange={c=>{setCenter(c);onSelect(c[0],c[1])}} drawing={drawing} onRoi={setRoi}/>
        <div className="absolute left-4 top-4 z-30 flex max-w-[min(520px,calc(100%-32px))] flex-wrap gap-2">
          <div className="glass flex items-center gap-2 rounded-lg px-3 py-2"><Search size={13} className="text-cyan-300"/><input value={query} onChange={e=>setQuery(e.target.value)} placeholder="Search sector…" className="w-44 bg-transparent text-xs text-white outline-none"/></div>
          <button onClick={()=>setDrawing(!drawing)} className={`btn btn--sm ${drawing ? "btn--primary" : "btn--secondary"}`}><Crosshair size={12}/>{drawing ? "Finish ROI" : "Draw ROI"}</button>
          <button onClick={()=>setSatellite(!satellite)} className="btn btn--secondary btn--sm"><Layers3 size={12}/>{satellite ? "Satellite" : "Base map"}</button>
        </div>
        <div className="absolute bottom-4 left-4 right-4 z-30 flex flex-wrap items-center gap-2">
          <div className="glass rounded-lg px-3 py-2 text-[9px] text-slate-500">CENTER <span className="font-mono text-cyan-200">{center[0].toFixed(4)}°, {center[1].toFixed(4)}°</span></div>
          <div className="glass flex items-center gap-2 rounded-lg px-3 py-2"><CalendarDays size={12} className="text-cyan-300"/><input type="range" min="2016" max="2026" value={year} onChange={e=>setYear(Number(e.target.value))}/><span className="font-mono text-xs text-cyan-200">{year}</span></div>
          {roi.length > 0 && <button className="btn btn--secondary btn--sm" onClick={()=>setRoi([])}>Clear ROI</button>}
        </div>
      </div>
      <aside className="relative z-20 min-h-0 overflow-y-auto border-l border-white/8 bg-[#05080d] p-4">
        <div className="text-[10px] font-semibold uppercase tracking-[.18em] text-slate-500">Quick sectors</div>
        <div className="mt-3 space-y-2">{filtered.map(s=><button key={s[0]} onClick={()=>{setCenter([s[1],s[2]]);onSelect(s[1],s[2])}} className="w-full rounded-lg border border-white/8 bg-white/[.025] p-3 text-left transition hover:border-cyan-300/30 hover:bg-cyan-300/5"><div className="text-xs font-medium text-slate-200">{s[0]}</div><div className="mt-1 text-[9px] text-slate-600">{s[3]}</div><div className="mt-2 font-mono text-[9px] text-cyan-300/60">{s[1].toFixed(4)}, {s[2].toFixed(4)}</div></button>)}</div>
        <div className="mt-5 border-t border-white/8 pt-4"><div className="text-[10px] font-semibold uppercase tracking-[.18em] text-slate-500">Temporal observation</div><div className="mt-3 rounded-lg border border-white/8 bg-white/[.025] p-3"><div className="flex justify-between text-xs"><span className="text-slate-300">Observation year</span><span className="font-mono text-cyan-200">{year}</span></div><div className="mt-2 text-[10px] leading-4 text-slate-600">Navigate to an AOI, draw an ROI, and return the spatial context to the SatQueryX mission pipeline.</div></div></div>
      </aside>
    </div>
  </div>;
}
