"use client";
import { useState } from "react";
import { BrainCircuit, Captions, GitCompareArrows, Radar, Send, Sparkles, Target } from "lucide-react";

const presets = [
  { label: "VQA", icon: BrainCircuit, query: "What major land-cover characteristics are visible in this scene?" },
  { label: "Caption", icon: Captions, query: "Describe this satellite scene and its dominant land-cover patterns." },
  { label: "Ground", icon: Target, query: "Locate and describe the most prominent target region in this scene." },
  { label: "Change", icon: GitCompareArrows, query: "Compare the two scenes and describe the major changes." },
  { label: "Optical + SAR", icon: Radar, query: "Compare the optical and SAR observations and explain the complementary information." },
];

export function QueryPanel({ query, onQuery, onRun, disabled, workflow, provider, onProvider }: { query: string; onQuery: (q:string)=>void; onRun: ()=>void; disabled:boolean; workflow:string; provider:"openrouter"|"gemini"; onProvider:(p:"openrouter"|"gemini")=>void; }) {
  const [precision, setPrecision] = useState(false);
  return <div className="panel p-3">
    <div className="panel-header mb-3"><div><h2 className="panel-title">Analysis Query</h2><p className="mt-1 text-[10px] text-slate-500">Natural language mission control</p></div><span className="badge badge--accent">{workflow}</span></div>
    <div className="mb-3 flex flex-wrap gap-2">{presets.map(({label,icon:Icon,query:q})=><button key={label} type="button" onClick={()=>onQuery(q)} className="btn btn--secondary btn--sm"><Icon size={12}/>{label}</button>)}</div>
    <textarea value={query} onChange={e=>onQuery(e.target.value)} onKeyDown={e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter')onRun()}} placeholder="Ask a spatial question about the active scene…" className="textarea min-h-[128px]" />
    <div className="mt-3 grid grid-cols-2 gap-2">
      <button type="button" onClick={()=>setPrecision(!precision)} className={`rounded-lg border px-3 py-2 text-left text-[10px] ${precision?"border-amber-300/30 bg-amber-300/10 text-amber-100":"border-white/8 bg-white/[.02] text-slate-500"}`}><Sparkles size={12} className="mr-1 inline"/>High-precision mode<div className="mt-1 text-[9px] text-slate-600">Escalation-ready workflow</div></button>
      <div className="rounded-lg border border-white/8 bg-white/[.02] p-2"><div className="text-[9px] uppercase tracking-[.14em] text-slate-600">Language layer</div><div className="mt-1 flex gap-2"><button type="button" onClick={()=>onProvider("openrouter")} className={`text-[10px] ${provider==='openrouter'?"text-cyan-200":"text-slate-500"}`}>OpenRouter</button><button type="button" onClick={()=>onProvider("gemini")} className={`text-[10px] ${provider==='gemini'?"text-cyan-200":"text-slate-500"}`}>Gemini</button></div></div>
    </div>
    <button type="button" disabled={disabled} onClick={onRun} className="btn btn--primary btn--full btn--lg mt-3"><Send size={15}/>{disabled?"Running mission…":"Run Analysis"}</button>
    <div className="mt-2 text-center text-[9px] text-slate-600">Ctrl/Cmd + Enter to execute · AI synthesis never overrides deterministic evidence.</div>
  </div>;
}
