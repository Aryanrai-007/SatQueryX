"use client";
import { AlertTriangle, CheckCircle2, Database, FileWarning, Target } from "lucide-react";

export function ResultPanel({ result, workflow, aiConfigured, model, error }: { result: any; workflow: string; aiConfigured: boolean | null; model: string; error: string }) {
  const payload = result?.result ?? result ?? {};
  const answer = result?.answer || payload?.answer || "";
  const blocked = result?.status === "blocked" || payload?.status === "REJECTED" || payload?.decision === "BLOCK";
  const regions = payload?.grounding || payload?.changed_regions || [];
  const evidence = payload?.evidence || [];
  const confidence = Number(payload?.confidence ?? 0);
  return <div className="panel flex min-h-0 flex-col p-3">
    <div className="panel-header mb-3"><div><h2 className="panel-title">Findings</h2><p className="mt-1 text-[10px] text-slate-500">Evidence-backed mission output</p></div><span className={`badge ${blocked?"badge--danger":answer?"badge--accent":"badge--warning"}`}>{blocked?"Blocked":answer?"Output ready":"Awaiting analysis"}</span></div>
    {!answer && !error ? <div className="empty-state"><Target className="empty-state__icon"/><div className="empty-state__text">Run a mission to populate findings</div><div className="empty-state__hint">Answer, evidence, confidence and validation state will appear here.</div></div> : <div className="min-h-0 flex-1 space-y-3 overflow-auto pr-1">
      {error && <div className="rounded-lg border border-red-400/20 bg-red-400/10 p-3 text-xs text-red-200"><AlertTriangle size={14} className="mr-2 inline"/>{error}</div>}
      {answer && <div className={`result-answer ${blocked?"border-red-400 bg-red-400/10 text-red-100":""}`}><div className="mb-2 text-[9px] uppercase tracking-[.18em] text-slate-500">Natural-language interpretation</div>{answer}</div>}
      {blocked && <div className="rounded-lg border border-red-400/20 bg-red-400/10 p-3 text-[11px] text-red-200"><FileWarning size={13} className="mr-2 inline"/>Analysis halted by validation. The language layer cannot override a failed spatial or data compatibility decision.</div>}
      {(confidence>0 || model) && <div className="rounded-lg border border-white/8 bg-white/[.025] p-3"><div className="flex items-center justify-between"><span className="text-[10px] uppercase tracking-[.16em] text-slate-500">Confidence / engine</span><span className="text-xs text-cyan-200">{confidence>0?`${(confidence*100).toFixed(0)}%`:"n/a"}</span></div><div className="mt-2 h-1.5 overflow-hidden rounded-full bg-white/8"><div className="h-full rounded-full bg-cyan-300" style={{width:`${Math.max(0,Math.min(100,confidence*100))}%`}}/></div><div className="mt-2 text-[10px] text-slate-500">{model || "Deterministic / local evidence layer"}</div></div>}
      {regions.length>0 && <div className="space-y-2"><div className="text-[10px] font-semibold uppercase tracking-[.16em] text-slate-500">Spatial evidence</div>{regions.slice(0,8).map((r:any,i:number)=><div key={i} className="rounded-lg border border-white/8 bg-white/[.025] p-2"><div className="flex items-center justify-between text-xs"><span className="text-slate-200">{r.label || r.change_type || `Region ${i+1}`}</span><span className="font-mono text-cyan-200">{r.confidence ? `${(r.confidence*100).toFixed(0)}%` : "evidence"}</span></div>{r.area_m2 && <div className="mt-1 text-[10px] text-slate-500">Area {(Number(r.area_m2)/10000).toFixed(2)} ha</div>}</div>)}</div>}
      {evidence.length>0 && <div><div className="mb-2 text-[10px] font-semibold uppercase tracking-[.16em] text-slate-500">Evidence trail</div><div className="space-y-2">{evidence.slice(0,7).map((e:any,i:number)=><div key={i} className="flex gap-2 rounded-lg border border-white/8 bg-white/[.02] p-2 text-[10px]"><CheckCircle2 size={13} className="mt-0.5 shrink-0 text-cyan-300"/><div><div className="text-slate-300">{e.step || e.title || `Evidence step ${i+1}`}</div>{e.confidence && <div className="mt-0.5 font-mono text-slate-600">confidence {(Number(e.confidence)*100).toFixed(0)}%</div>}</div></div>)}</div></div>}
      <div className="rounded-lg border border-white/8 bg-white/[.02] p-2 text-[10px] text-slate-500"><Database size={12} className="mr-2 inline text-cyan-300"/>Workflow: <span className="text-slate-300">{workflow}</span> · AI configured: <span className={aiConfigured?"text-cyan-200":"text-amber-200"}>{aiConfigured===null?"unknown":String(aiConfigured)}</span></div>
    </div>}
  </div>;
}
