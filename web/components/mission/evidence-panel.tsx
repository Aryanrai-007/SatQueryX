"use client";
import { CheckCircle2, FileText, Gauge, Hash, Layers3 } from "lucide-react";

export function EvidencePanel({ result, workflow, changeStats }: { result:any; workflow:string; changeStats:{changedFraction:number;meanDifference:number}|null }) {
  const payload = result?.result ?? result ?? {};
  const evidence = Array.isArray(payload?.evidence) ? payload.evidence : [];
  const model = payload?.model_used || "Not configured / not executed";
  return <div className="panel p-3">
    <div className="panel-header mb-3"><div><h2 className="panel-title">Evidence & Traceability</h2><p className="mt-1 text-[10px] text-slate-500">What the system actually observed</p></div><span className="badge badge--accent">{workflow}</span></div>
    <div className="evidence-list">
      {evidence.length ? evidence.map((e:any,i:number)=><div key={i} className="evidence-card"><div className="flex items-center gap-2 text-[10px] uppercase tracking-[.14em] text-slate-500"><CheckCircle2 size={12} className="text-cyan-300"/>Step {i+1}</div><div className="evidence-card__desc text-slate-300">{e.step || e.title || "Evidence step"}</div><div className="font-mono text-[10px] text-slate-600">{e.confidence ? `confidence ${(Number(e.confidence)*100).toFixed(0)}%` : "recorded observation"}</div></div>) : <div className="evidence-card"><Layers3 size={15} className="text-cyan-300"/><div className="text-xs text-slate-300">Mission trace will populate after execution.</div><div className="text-[10px] text-slate-600">No evidence is invented to fill an empty result.</div></div>}
    </div>
    <div className="mt-3 grid gap-2 md:grid-cols-4">
      <div className="rounded-lg border border-white/8 bg-white/[.02] p-2"><Gauge size={12} className="text-cyan-300"/><div className="mt-1 text-[9px] uppercase tracking-[.14em] text-slate-600">Engine</div><div className="mt-1 truncate text-[10px] text-slate-300">{model}</div></div>
      <div className="rounded-lg border border-white/8 bg-white/[.02] p-2"><FileText size={12} className="text-cyan-300"/><div className="mt-1 text-[9px] uppercase tracking-[.14em] text-slate-600">Change</div><div className="mt-1 text-[10px] text-slate-300">{changeStats ? `${(changeStats.changedFraction*100).toFixed(2)}% changed` : "Not computed"}</div></div>
      <div className="rounded-lg border border-white/8 bg-white/[.02] p-2"><Hash size={12} className="text-cyan-300"/><div className="mt-1 text-[9px] uppercase tracking-[.14em] text-slate-600">Audit</div><div className="mt-1 font-mono text-[10px] text-slate-300">Mission trace</div></div>
      <div className="rounded-lg border border-white/8 bg-white/[.02] p-2"><Layers3 size={12} className="text-cyan-300"/><div className="mt-1 text-[9px] uppercase tracking-[.14em] text-slate-600">Mean delta</div><div className="mt-1 font-mono text-[10px] text-slate-300">{changeStats ? changeStats.meanDifference.toFixed(4) : "—"}</div></div>
    </div>
  </div>;
}
