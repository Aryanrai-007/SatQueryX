"use client";
import { useState } from "react";
import { X, ShieldCheck } from "lucide-react";

export function AuditModal({ open, onClose, entries }: { open:boolean; onClose:()=>void; entries:any[] }) {
  const [selected, setSelected] = useState(0);
  if (!open) return null;
  return <div className="audit-overlay" onClick={onClose} role="dialog" aria-modal="true">
    <div className="panel audit-modal" onClick={e=>e.stopPropagation()}>
      <div className="modal-header mb-4 border-b border-white/8 pb-3"><div className="flex items-center gap-2"><ShieldCheck size={16} className="text-cyan-300"/><div><div className="panel-title">Execution Audit Trail</div><div className="mt-1 text-[10px] text-slate-500">Mission-local trace of routing, evidence and model state</div></div></div><button className="btn btn--ghost btn--sm" onClick={onClose}><X size={14}/></button></div>
      {!entries.length ? <div className="empty-state"><div className="empty-state__text">No executions recorded in this session.</div></div> : <div className="grid gap-4 lg:grid-cols-[1fr_1.2fr]"><div className="space-y-1">{entries.map((entry:any,i:number)=><button key={i} onClick={()=>setSelected(i)} className={`w-full rounded-lg border p-3 text-left ${i===selected?"border-cyan-300/30 bg-cyan-300/10":"border-white/8 bg-white/[.02]"}`}><div className="text-[10px] uppercase tracking-[.14em] text-slate-600">{entry.time}</div><div className="mt-1 text-xs text-slate-200">{entry.workflow}</div><div className="mt-1 truncate text-[10px] text-slate-500">{entry.query}</div></button>)}</div><div className="rounded-lg border border-white/8 bg-white/[.02] p-3"><div className="text-[10px] uppercase tracking-[.14em] text-slate-600">Execution record</div><pre className="mt-3 max-h-[55vh] overflow-auto whitespace-pre-wrap text-[10px] leading-5 text-slate-300">{JSON.stringify(entries[selected],null,2)}</pre></div></div>}
    </div>
  </div>;
}
