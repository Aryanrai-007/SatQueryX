"use client";
import { CheckCircle2, AlertTriangle, CircleDot } from "lucide-react";
import type { Step } from "./types";

export function StepList({ steps }: { steps: Step[] }) {
  return (
    <div className="space-y-1">
      {steps.map((step, index) => (
        <div key={step.id} className="flex gap-3">
          <div className="flex flex-col items-center">
            <div className={`mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border ${
              step.status === "done" ? "border-cyan-300/30 bg-cyan-300/10 text-cyan-200" :
              step.status === "running" ? "border-amber-300/30 bg-amber-300/10 text-amber-200 animate-pulse" :
              step.status === "error" ? "border-red-300/30 bg-red-300/10 text-red-200" :
              step.status === "waiting" ? "border-violet-300/30 bg-violet-300/10 text-violet-200" :
              "border-white/10 bg-white/[0.02] text-slate-600"
            }`}>
              {step.status === "done" ? <CheckCircle2 size={14}/> : step.status === "error" ? <AlertTriangle size={14}/> : step.status === "waiting" ? <CircleDot size={14}/> : <span className="text-[9px]">{index + 1}</span>}
            </div>
            {index < steps.length - 1 && <div className="h-full min-h-5 w-px bg-white/8"/>}
          </div>
          <div className="pb-3">
            <div className="text-xs font-medium text-slate-200">{step.title}</div>
            <div className="text-[10px] leading-4 text-slate-500">{step.detail}</div>
          </div>
        </div>
      ))}
    </div>
  );
}
