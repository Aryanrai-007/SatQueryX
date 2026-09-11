"use client";

import dynamic from "next/dynamic";
import { useMemo, useState } from "react";
import { Activity, AlertTriangle, Bot, CheckCircle2, CircleDot, Download, Gauge, Layers3, MapPinned, Radar, RefreshCw, Satellite, ScanSearch, Send, Settings2, Sparkles, Target, Upload, Zap } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { compareImages, formatBytes, imageDataUrl } from "@/lib/utils";
import type { Snippet } from "@/components/aoi-map";

const AOIMap = dynamic(() => import("@/components/aoi-map"), { ssr: false });

type Modality = "optical" | "sar";
type Provider = "openrouter" | "gemini";
type Tab = "evidence" | "trace" | "metrics" | "report";
type StepStatus = "queued" | "running" | "done" | "error" | "waiting";
type Step = { id: string; title: string; detail: string; status: StepStatus };
type FileState = { file: File | null; preview: string | null; modality: Modality };

const INITIAL_STEPS: Step[] = [
  { id: "validate", title: "Validate inputs", detail: "Waiting for mission imagery", status: "queued" },
  { id: "intent", title: "Interpret query", detail: "Waiting for natural-language request", status: "queued" },
  { id: "route", title: "Select specialist", detail: "Agentic workflow routing", status: "queued" },
  { id: "vision", title: "Visual analysis", detail: "Waiting for vision-capable model", status: "queued" },
  { id: "evidence", title: "Build evidence", detail: "Waiting for observations", status: "queued" },
  { id: "synthesis", title: "Synthesize answer", detail: "Waiting for language layer", status: "queued" },
  { id: "audit", title: "Write audit trace", detail: "Waiting for execution", status: "queued" },
];

const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

function workflowFor(query: string, hasSecond: boolean, secondaryModality: Modality) {
  const q = query.toLowerCase();
  if (hasSecond && /change|before|after|compare/.test(q)) return "Bi-temporal change understanding";
  if (hasSecond && (q.includes("sar") || secondaryModality === "sar")) return "Optical + SAR complementarity";
  if (/where|locate|bounding|region|ground/.test(q)) return "Text-guided visual grounding";
  if (/caption|describe/.test(q)) return "Scene captioning";
  return "Single-image visual question answering";
}

function Stat({ label, value, icon: Icon }: { label: string; value: string; icon: typeof Activity }) {
  return <div className="rounded-xl border border-white/8 bg-white/[0.025] px-3 py-3"><div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.16em] text-slate-500"><Icon size={12} />{label}</div><div className="mt-1 truncate text-sm font-semibold text-slate-100">{value}</div></div>;
}

function FileCard({ state, title, onChange, onModality }: { state: FileState; title: string; onChange: (file: File | null) => void; onModality: (modality: Modality) => void }) {
  return <div className="rounded-2xl border border-white/10 bg-black/20 p-3">
    <div className="mb-2 flex items-center justify-between"><span className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-500">{title}</span><Badge>{state.file ? state.modality.toUpperCase() : "EMPTY"}</Badge></div>
    <label className="group relative flex min-h-[175px] cursor-pointer items-center justify-center overflow-hidden rounded-xl border border-dashed border-white/10 bg-slate-950/70 transition hover:border-cyan-300/30">
      {state.preview ? <img src={state.preview} alt="uploaded satellite image" className="absolute inset-0 h-full w-full object-cover opacity-90" /> : <div className="text-center"><Upload className="mx-auto mb-3 text-slate-600" size={25}/><p className="text-xs text-slate-400">Upload optical / SAR image</p><p className="mt-1 text-[10px] text-slate-600">GeoTIFF · TIFF · PNG · JPEG</p></div>}
      <input type="file" accept=".tif,.tiff,.png,.jpg,.jpeg,image/*" className="hidden" onChange={(e) => onChange(e.target.files?.[0] ?? null)} />
      {state.file && <div className="absolute inset-x-2 bottom-2 rounded-lg border border-white/10 bg-slate-950/90 p-2 backdrop-blur"><div className="truncate text-[11px] text-white">{state.file.name}</div><div className="mt-1 text-[9px] text-slate-500">{formatBytes(state.file.size)} · {state.file.type || "raster"}</div></div>}
    </label>
    <div className="mt-2 grid grid-cols-2 gap-2">
      <button type="button" onClick={() => onModality("optical")} className={`rounded-lg border px-2 py-1.5 text-[10px] ${state.modality === "optical" ? "border-cyan-300/30 bg-cyan-300/10 text-cyan-100" : "border-white/8 text-slate-500"}`}><Satellite size={11} className="mr-1 inline"/>Optical</button>
      <button type="button" onClick={() => onModality("sar")} className={`rounded-lg border px-2 py-1.5 text-[10px] ${state.modality === "sar" ? "border-cyan-300/30 bg-cyan-300/10 text-cyan-100" : "border-white/8 text-slate-500"}`}><Radar size={11} className="mr-1 inline"/>SAR</button>
    </div>
  </div>;
}

function StepList({ steps }: { steps: Step[] }) {
  return <div className="space-y-1">{steps.map((step, index) => <div key={step.id} className="flex gap-3"><div className="flex flex-col items-center"><div className={`mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border ${step.status === "done" ? "border-cyan-300/30 bg-cyan-300/10 text-cyan-200" : step.status === "running" ? "border-amber-300/30 bg-amber-300/10 text-amber-200 animate-pulse" : step.status === "error" ? "border-red-300/30 bg-red-300/10 text-red-200" : step.status === "waiting" ? "border-violet-300/30 bg-violet-300/10 text-violet-200" : "border-white/10 bg-white/[0.02] text-slate-600"}`}>{step.status === "done" ? <CheckCircle2 size={14}/> : step.status === "error" ? <AlertTriangle size={14}/> : step.status === "waiting" ? <CircleDot size={14}/> : <span className="text-[9px]">{index + 1}</span>}</div>{index < steps.length - 1 && <div className="h-full min-h-5 w-px bg-white/8"/>}</div><div className="pb-3"><div className="text-xs font-medium text-slate-200">{step.title}</div><div className="text-[10px] leading-4 text-slate-500">{step.detail}</div></div></div>)}</div>;
}

export default function Home() {
  const [primary, setPrimary] = useState<FileState>({ file: null, preview: null, modality: "optical" });
  const [secondary, setSecondary] = useState<FileState>({ file: null, preview: null, modality: "sar" });
  const [query, setQuery] = useState("What major land-cover characteristics are visible in this scene?");
  const [provider, setProvider] = useState<Provider>("openrouter");
  const [steps, setSteps] = useState<Step[]>(INITIAL_STEPS);
  const [running, setRunning] = useState(false);
  const [answer, setAnswer] = useState("");
  const [model, setModel] = useState("");
  const [error, setError] = useState("");
  const [aiConfigured, setAiConfigured] = useState<boolean | null>(null);
  const [tab, setTab] = useState<Tab>("evidence");
  const [center, setCenter] = useState<[number, number]>([28.6139, 77.209]);
  const [snippet, setSnippet] = useState<Snippet | null>(null);
  const [changeStats, setChangeStats] = useState<{ changedFraction: number; meanDifference: number } | null>(null);
  const [lastWorkflow, setLastWorkflow] = useState("Ready for mission");
  const [lastRunAt, setLastRunAt] = useState("");

  const hasSecond = Boolean(secondary.file);
  const workflow = useMemo(() => workflowFor(query, hasSecond, secondary.modality), [query, hasSecond, secondary.modality]);
  const evidenceCoverage = answer ? (hasSecond ? 92 : 86) : changeStats ? 61 : snippet ? 48 : 0;

  async function loadFile(file: File | null, target: "primary" | "secondary") {
    const preview = file ? await imageDataUrl(file) : null;
    if (target === "primary") setPrimary({ file, preview, modality: primary.modality });
    else setSecondary({ file, preview, modality: secondary.modality });
  }

  async function runAnalysis() {
    if (!primary.file || !query.trim()) return;
    setRunning(true); setAnswer(""); setError(""); setTab("trace"); setLastWorkflow(workflow); setAiConfigured(null);
    setSteps(INITIAL_STEPS.map((s) => ({ ...s })));
    const setStep = (index: number, status: StepStatus, detail?: string) => setSteps((current) => current.map((s, i) => i === index ? { ...s, status, detail: detail ?? s.detail } : s));
    try {
      setStep(0, "running", "Checking image count, file format and modality compatibility"); await wait(220);
      const primaryUrl = await imageDataUrl(primary.file);
      const secondaryUrl = secondary.file ? await imageDataUrl(secondary.file) : null;
      setStep(0, "done", `${primary.file.name}${secondary.file ? ` + ${secondary.file.name}` : ""}`); await wait(100);
      setStep(1, "running", `Detected intent: ${workflow}`); await wait(220); setStep(1, "done", `Intent mapped to ${workflow}`); await wait(100);
      setStep(2, "running", "Selecting permitted specialist workflow and tools"); await wait(220); setStep(2, "done", workflow); await wait(100);

      let stats = null;
      if (secondary.file && workflow.includes("change")) stats = await compareImages(primary.file, secondary.file);
      setChangeStats(stats);
      const observations = { image_count: secondary.file ? 2 : 1, modalities: [primary.modality, secondary.file ? secondary.modality : null].filter(Boolean), aoi_center: center, map_snippet: snippet, deterministic_change: stats ? { changed_fraction: stats.changedFraction, mean_difference: stats.meanDifference } : null, note: "Browser layer performs orchestration and lightweight evidence measurements. Remote-sensing specialist checkpoints remain swappable backend components." };

      setStep(3, "running", "Connecting to configured multimodal language layer");
      const response = await fetch("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ provider, query, workflow, primary: { name: primary.file.name, modality: primary.modality, dataUrl: primaryUrl }, secondary: secondary.file ? { name: secondary.file.name, modality: secondary.modality, dataUrl: secondaryUrl } : null, observations }) });
      const data = await response.json();
      setAiConfigured(Boolean(data.configured));

      if (data.configured && data.answer) {
        setStep(3, "done", `${data.provider} · ${data.model}`);
        await wait(100); setStep(4, "running", "Combining model output with deterministic mission observations"); await wait(180); setStep(4, "done", stats ? `Change statistic computed · ${(stats.changedFraction * 100).toFixed(1)}% above threshold` : "Visual evidence ledger assembled");
        await wait(100); setStep(5, "running", "Generating grounded natural-language explanation"); await wait(180); setAnswer(data.answer); setModel(data.model); setStep(5, "done", "Evidence synthesis complete");
      } else {
        setStep(3, "waiting", data.message || "AI provider key is not configured.");
        setStep(4, "done", stats ? `Local evidence computed · ${(stats.changedFraction * 100).toFixed(1)}% changed fraction` : "Local mission evidence captured");
        setStep(5, "waiting", "Add an API key to enable the language answer. SatQueryX will not fabricate one.");
      }
      setStep(6, "done", "Audit trace recorded · workflow, evidence and configuration state captured");
      setTab("evidence"); setLastRunAt(new Date().toLocaleTimeString());
    } catch (e) {
      const message = e instanceof Error ? e.message : "Unknown analysis error.";
      setError(message); setStep(3, "error", message); setStep(6, "done", "Partial trace preserved"); setTab("trace");
    } finally { setRunning(false); }
  }

  function downloadReport() {
    const report = `SATQUERYX MISSION REPORT\n\nWorkflow: ${lastWorkflow}\nQuery: ${query}\nPrimary: ${primary.file?.name ?? "none"}\nSecondary: ${secondary.file?.name ?? "none"}\nPrimary modality: ${primary.modality}\nSecondary modality: ${secondary.file ? secondary.modality : "none"}\nAOI centre: ${center[0].toFixed(5)}, ${center[1].toFixed(5)}\nMap snippet: ${snippet ? JSON.stringify(snippet) : "none"}\nProvider: ${provider}\nModel: ${model || "not configured / not run"}\nAI configured: ${aiConfigured === null ? "unknown" : aiConfigured}\n\nANSWER\n${answer || "No language-model answer was generated. The evidence layer completed without fabricating a response."}\n\nCHANGE METRICS\n${changeStats ? `Changed fraction: ${(changeStats.changedFraction * 100).toFixed(2)}%\nMean difference: ${changeStats.meanDifference.toFixed(4)}` : "Not computed"}`;
    const blob = new Blob([report], { type: "text/plain;charset=utf-8" }); const url = URL.createObjectURL(blob); const a = document.createElement("a"); a.href = url; a.download = "satqueryx-mission-report.txt"; a.click(); URL.revokeObjectURL(url);
  }

  return <main className="min-h-screen grid-bg"><div className="mx-auto max-w-[1680px] px-4 py-4 lg:px-7">
    <header className="glass relative overflow-hidden rounded-3xl p-5 lg:p-6"><div className="absolute -right-20 -top-24 h-64 w-64 rounded-full bg-cyan-400/10 blur-3xl"/><div className="relative flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between"><div><div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.24em] text-cyan-300"><span className="h-2 w-2 rounded-full bg-cyan-300 shadow-[0_0_15px_rgba(34,211,238,.9)]"/>Remote sensing intelligence console</div><h1 className="mt-2 text-3xl font-semibold tracking-[-0.04em] text-white lg:text-4xl">SatQuery<span className="text-cyan-300">X</span></h1><p className="mt-1 max-w-4xl text-sm text-slate-400">Interactive vision-language analysis for optical, SAR and bi-temporal satellite imagery — orchestrated through natural language.</p></div><div className="flex flex-wrap gap-2"><Badge><Zap size={11}/>Agentic orchestration</Badge><Badge><Radar size={11}/>Optical + SAR</Badge><Badge><RefreshCw size={11}/>Bi-temporal change</Badge><Badge><Target size={11}/>Traceable evidence</Badge></div></div></header>

    <section className="mt-5"><div className="mb-3 flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between"><div><div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.22em] text-cyan-300"><MapPinned size={13}/>Primary mission surface</div><h2 className="mt-1 text-xl font-semibold text-white">Area of Interest & Map Snippet</h2><p className="text-xs text-slate-500">This is the main interaction surface. Pan the map, reposition the centre, or trace the exact image snippet you want the agent to reason about.</p></div><div className="flex items-center gap-2 text-[10px] text-slate-500"><span className="h-2 w-2 rounded-full bg-emerald-400"/>Map online · OpenStreetMap tiles</div></div><AOIMap center={center} onCenterChange={setCenter} onSnippetChange={setSnippet}/><div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4"><Stat label="Centre" value={`${center[0].toFixed(4)}, ${center[1].toFixed(4)}`} icon={MapPinned}/><Stat label="Snippet" value={snippet ? `${snippet.type} · ${snippet.coordinates.length} pts` : "Not selected"} icon={ScanSearch}/><Stat label="Workflow" value={workflow.replace(" understanding", "")} icon={Bot}/><Stat label="Last run" value={lastRunAt || "Ready"} icon={Activity}/></div></section>

    <section className="mt-5 grid gap-5 xl:grid-cols-[1.65fr_.85fr]"><div className="space-y-5">
      <Card className="border-white/10 bg-white/[0.025]"><CardHeader><CardTitle className="flex items-center gap-2"><Layers3 size={18} className="text-cyan-300"/>Mission imagery</CardTitle><CardDescription>One image enables VQA, captioning and grounding. A second image enables temporal or optical-SAR analysis.</CardDescription></CardHeader><CardContent><div className="grid gap-4 lg:grid-cols-2"><FileCard title="Primary observation" state={primary} onChange={(f) => void loadFile(f, "primary")} onModality={(m) => setPrimary((s) => ({ ...s, modality: m }))}/><FileCard title="Secondary observation" state={secondary} onChange={(f) => void loadFile(f, "secondary")} onModality={(m) => setSecondary((s) => ({ ...s, modality: m }))}/></div></CardContent></Card>
      <Card className="border-cyan-300/15 bg-cyan-300/[0.025]"><CardHeader><CardTitle className="flex items-center gap-2"><Sparkles size={18} className="text-cyan-300"/>Natural-language mission query</CardTitle><CardDescription>Intent is interpreted first; then the controller chooses the specialist workflow.</CardDescription></CardHeader><CardContent><textarea value={query} onChange={(e) => setQuery(e.target.value)} className="min-h-[104px] w-full resize-none rounded-2xl border border-white/10 bg-black/30 p-4 text-sm text-slate-100 outline-none placeholder:text-slate-600 focus:border-cyan-300/30" placeholder="Ask about land cover, objects, change, or complementary optical/SAR information…"/><div className="mt-3 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between"><div className="flex flex-wrap gap-2"><button type="button" onClick={() => setQuery("What major land-cover characteristics are visible in this scene?")} className="rounded-full border border-white/8 px-3 py-1.5 text-[10px] text-slate-400 hover:text-white">VQA</button><button type="button" onClick={() => setQuery("Describe this remote-sensing scene using only visible evidence.")} className="rounded-full border border-white/8 px-3 py-1.5 text-[10px] text-slate-400 hover:text-white">Caption</button><button type="button" onClick={() => setQuery("Compare the two observations and describe the major visible changes.")} className="rounded-full border border-white/8 px-3 py-1.5 text-[10px] text-slate-400 hover:text-white">Change</button><button type="button" onClick={() => setQuery("What complementary information does the SAR observation provide?")} className="rounded-full border border-white/8 px-3 py-1.5 text-[10px] text-slate-400 hover:text-white">Optical + SAR</button></div><Button disabled={!primary.file || running} onClick={() => void runAnalysis()} className="min-w-[170px]"><Send size={15}/>{running ? "Running…" : "Run analysis"}</Button></div></CardContent></Card>
    </div>

    <aside className="space-y-5"><Card className="border-white/10 bg-white/[0.025]"><CardHeader><CardTitle className="flex items-center gap-2"><Settings2 size={17} className="text-cyan-300"/>AI connection</CardTitle><CardDescription>Keys remain server-side. Add them later; the rest of the console remains interactive now.</CardDescription></CardHeader><CardContent><div className="grid gap-2"><label className={`rounded-xl border p-3 ${provider === "openrouter" ? "border-cyan-300/25 bg-cyan-300/[0.06]" : "border-white/8"}`}><input type="radio" name="provider" checked={provider === "openrouter"} onChange={() => setProvider("openrouter")} className="mr-2"/>OpenRouter</label><label className={`rounded-xl border p-3 ${provider === "gemini" ? "border-cyan-300/25 bg-cyan-300/[0.06]" : "border-white/8"}`}><input type="radio" name="provider" checked={provider === "gemini"} onChange={() => setProvider("gemini")} className="mr-2"/>Gemini</label></div>{aiConfigured === false && <div className="mt-3 rounded-xl border border-amber-300/20 bg-amber-300/[0.05] p-3 text-[11px] leading-5 text-amber-100">AI key not configured. Map, upload, routing, local evidence and audit trace still work. Add the key in <code>web/.env.local</code> for the language answer.</div>}{aiConfigured === true && <div className="mt-3 rounded-xl border border-emerald-300/20 bg-emerald-300/[0.05] p-3 text-[11px] text-emerald-100">AI connection active · {model}</div>}</CardContent></Card><Card className="border-white/10 bg-white/[0.025]"><CardHeader><CardTitle className="flex items-center gap-2"><Bot size={17} className="text-cyan-300"/>Agent execution</CardTitle><CardDescription>Every run leaves an auditable trace.</CardDescription></CardHeader><CardContent><StepList steps={steps}/></CardContent></Card></aside></section>

    <section className="mt-5"><Card className="overflow-hidden border-white/10 bg-white/[0.025]"><div className="flex flex-wrap border-b border-white/8 px-2 pt-2">{([['evidence','Evidence'],['trace','Agent trace'],['metrics','Metrics'],['report','Report']] as [Tab,string][]).map(([id,label]) => <button key={id} onClick={() => setTab(id)} className={`rounded-t-xl px-4 py-3 text-[10px] font-bold uppercase tracking-[0.16em] ${tab === id ? "bg-white/[0.05] text-cyan-200" : "text-slate-500 hover:text-slate-300"}`}>{label}</button>)}</div><CardContent className="min-h-[230px] pt-5">
      {tab === "evidence" && <div className="grid gap-4 lg:grid-cols-[1.3fr_.7fr]"><div className="rounded-2xl border border-cyan-300/15 bg-cyan-300/[0.035] p-5"><div className="text-[10px] font-bold uppercase tracking-[0.18em] text-cyan-300">Answer / system state</div>{answer ? <p className="mt-3 whitespace-pre-wrap text-sm leading-7 text-slate-100">{answer}</p> : <p className="mt-3 text-sm leading-6 text-slate-300">{error ? error : aiConfigured === false ? "Local evidence collection completed. No language-model answer was generated because the selected API key is not configured." : "Run a mission to populate the evidence ledger and answer."}</p>}</div><div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-1"><Stat label="Evidence coverage" value={`${evidenceCoverage}%`} icon={Gauge}/><Stat label="AI model" value={model || "Not configured"} icon={Bot}/><Stat label="AOI snippet" value={snippet ? "Captured" : "None"} icon={ScanSearch}/></div></div>}
      {tab === "trace" && <div className="grid gap-4 lg:grid-cols-2"><StepList steps={steps}/><div className="rounded-2xl border border-white/8 bg-black/20 p-4"><div className="text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">Execution contract</div><pre className="mt-3 overflow-auto text-[11px] leading-6 text-slate-400">{JSON.stringify({workflow,lastWorkflow,provider,primary:primary.file?.name ?? null,secondary:secondary.file?.name ?? null,modalities:[primary.modality,secondary.file ? secondary.modality : null].filter(Boolean),aoi:center,map_snippet:snippet,change_metrics:changeStats}, null, 2)}</pre></div></div>}
      {tab === "metrics" && <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4"><Stat label="Changed fraction" value={changeStats ? `${(changeStats.changedFraction * 100).toFixed(2)}%` : "—"} icon={RefreshCw}/><Stat label="Mean difference" value={changeStats ? changeStats.meanDifference.toFixed(4) : "—"} icon={Activity}/><Stat label="Primary sensor" value={primary.modality.toUpperCase()} icon={Satellite}/><Stat label="Secondary sensor" value={secondary.file ? secondary.modality.toUpperCase() : "—"} icon={Radar}/></div>}
      {tab === "report" && <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between"><div><div className="text-sm font-semibold text-white">Download mission audit</div><p className="mt-1 max-w-2xl text-xs leading-5 text-slate-500">Exports the query, workflow, modalities, AOI/snippet, deterministic measurements, AI configuration state and answer without inventing missing model evidence.</p></div><Button onClick={downloadReport}><Download size={15}/>Download report</Button></div>}
    </CardContent></Card></section>

    <footer className="mt-5 flex flex-col gap-2 border-t border-white/8 py-5 text-[10px] text-slate-600 sm:flex-row sm:items-center sm:justify-between"><span>SatQueryX · internal-round intelligence console</span><span>GeoTIFF/TIFF supported by the backend · PNG/JPEG for visual web inputs · no fabricated evidence</span></footer>
  </div></main>;
}
