import { Activity, Database, MapPinned, ShieldCheck } from "lucide-react";

export function StatusStrip({ workflow, files, ai, lastRun }: { workflow:string; files:number; ai:boolean|null; lastRun:string }) {
  const state = ai===true ? "AI CONNECTED" : ai===false ? "AI PENDING" : "READY";
  return <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
    <div className="glass rounded-xl px-3 py-2"><div className="flex items-center gap-2 text-[9px] uppercase tracking-[.16em] text-slate-600"><Activity size={11}/>Pipeline</div><div className="mt-1 truncate text-[11px] text-slate-300">{workflow}</div></div>
    <div className="glass rounded-xl px-3 py-2"><div className="flex items-center gap-2 text-[9px] uppercase tracking-[.16em] text-slate-600"><Database size={11}/>Imagery</div><div className="mt-1 text-[11px] text-slate-300">{files} scene{files===1?'':'s'} loaded</div></div>
    <div className="glass rounded-xl px-3 py-2"><div className="flex items-center gap-2 text-[9px] uppercase tracking-[.16em] text-slate-600"><ShieldCheck size={11}/>Language layer</div><div className={`mt-1 text-[11px] ${ai===true?'text-cyan-200':ai===false?'text-amber-200':'text-slate-300'}`}>{state}</div></div>
    <div className="glass rounded-xl px-3 py-2"><div className="flex items-center gap-2 text-[9px] uppercase tracking-[.16em] text-slate-600"><MapPinned size={11}/>Last execution</div><div className="mt-1 font-mono text-[11px] text-slate-300">{lastRun||"—"}</div></div>
  </div>
}
