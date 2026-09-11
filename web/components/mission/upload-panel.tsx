"use client";
import { useRef, useState } from "react";
import { Radar, Satellite, Upload } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { FileState, Modality } from "./types";

export function UploadPanel({
  primary, secondary, onPrimary, onSecondary, onPrimaryModality, onSecondaryModality,
}: {
  primary: FileState; secondary: FileState; onPrimary: (file: File | null) => void; onSecondary: (file: File | null) => void;
  onPrimaryModality: (m: Modality) => void; onSecondaryModality: (m: Modality) => void;
}) {
  const pRef = useRef<HTMLInputElement>(null); const sRef = useRef<HTMLInputElement>(null); const [preset, setPreset] = useState("");
  const presets = [["single-optical","Optical"],["single-sar","SAR Radar"],["change","Change T0 → T1"],["fusion","Optical + SAR"]];
  const apply = (id:string) => { setPreset(id); if(id==="single-optical"){onSecondary(null);onPrimaryModality("optical")} if(id==="single-sar"){onSecondary(null);onPrimaryModality("sar")} if(id==="change"){onPrimaryModality("optical");onSecondaryModality("optical")} if(id==="fusion"){onPrimaryModality("optical");onSecondaryModality("sar")} };
  const SceneCard = ({state,title,refEl,onFile,modality,onModality,hint}:{state:FileState;title:string;refEl:React.RefObject<HTMLInputElement|null>;onFile:(f:File|null)=>void;modality:Modality;onModality:(m:Modality)=>void;hint:string}) => <div className="upload-zone">
    <div className="upload-zone__header"><h3 className="upload-zone__title">{title}</h3><Badge>{state.file?modality.toUpperCase():"EMPTY"}</Badge></div>
    <label className="group relative flex min-h-[148px] cursor-pointer items-center justify-center overflow-hidden rounded-lg border border-dashed border-white/10 bg-black/20 hover:border-cyan-300/30">
      {state.preview?<img src={state.preview} alt="satellite upload" className="absolute inset-0 h-full w-full object-cover opacity-90"/>:<div className="text-center"><Upload className="mx-auto mb-3 text-slate-600" size={22}/><p className="text-xs text-slate-400">Select satellite raster</p><p className="mt-1 text-[10px] text-slate-600">GeoTIFF · TIFF · PNG · JPEG</p></div>}
      <input ref={refEl} className="hidden" type="file" accept=".tif,.tiff,.png,.jpg,.jpeg,image/*" onChange={e=>onFile(e.target.files?.[0]??null)}/>
      {state.file&&<div className="absolute inset-x-2 bottom-2 rounded border border-white/10 bg-slate-950/90 p-2 text-[10px]"><div className="truncate text-white">{state.file.name}</div><div className="mt-1 text-slate-500">{(state.file.size/1024).toFixed(0)} KB · {state.file.type||"raster"}</div></div>}
    </label>
    <div className="mt-2 grid grid-cols-2 gap-2"><button type="button" onClick={()=>onModality("optical")} className={`rounded border px-2 py-1.5 text-[10px] ${modality==="optical"?"border-cyan-300/30 bg-cyan-300/10 text-cyan-100":"border-white/8 text-slate-500"}`}><Satellite size={11} className="mr-1 inline"/>Optical</button><button type="button" onClick={()=>onModality("sar")} className={`rounded border px-2 py-1.5 text-[10px] ${modality==="sar"?"border-cyan-300/30 bg-cyan-300/10 text-cyan-100":"border-white/8 text-slate-500"}`}><Radar size={11} className="mr-1 inline"/>SAR</button></div>
    <div className="mt-2 text-[10px] text-slate-600">{hint}</div>
  </div>;
  return <div className="panel p-3"><div className="panel-header mb-3"><div><h2 className="panel-title">Input</h2><p className="mt-1 text-[10px] text-slate-500">Mission imagery & modality</p></div><span className="badge badge--success">Ready</span></div><div className="preset-section mb-3"><div className="preset-label">Workflow presets</div><div className="preset-grid">{presets.map(([id,label])=><button key={id} className={`preset-btn ${preset===id?"is-active":""}`} onClick={()=>apply(id)}>{label}</button>)}</div></div><div className="space-y-3"><SceneCard state={primary} title="Primary / T0" refEl={pRef} onFile={onPrimary} modality={primary.modality} onModality={onPrimaryModality} hint="Required for every analysis"/><SceneCard state={secondary} title="Secondary / T1 or SAR" refEl={sRef} onFile={onSecondary} modality={secondary.modality} onModality={onSecondaryModality} hint="Used for change or cross-modal workflows"/></div></div>;
}
