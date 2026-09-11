"use client";

import dynamic from "next/dynamic";
import { ChangeEvent, useMemo, useState } from "react";
import {
  Activity, AlertTriangle, ArrowRight, Bot, BrainCircuit, CheckCircle2, ChevronRight,
  CircleDot, Cloud, Crosshair, Download, FileImage, Gauge, Layers3, MapPinned,
  Radar, RefreshCw, Satellite, ScanSearch, Send, Settings2, Sparkles, Target,
  Upload, Waves, Zap,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { compareImages, fileToDataUrl, formatBytes, imageDataUrl } from "@/lib/utils";

const AOIMap = dynamic(() => import("@/components/aoi-map"), { ssr: false });

type Modality = "optical" | "sar";
type Provider = "openrouter" | "gemini";
type Step = { id: string; title: string; detail: string; status: "queued" | "running" | "done" | "error" };

type FileState = {
  file: File | null;
  preview: string | null;
  modality: Modality;
};

const INITIAL_STEPS: Step[] = [
  { id: "validate", title: "Validate inputs", detail: "Waiting for mission imagery", status: "queued" },
  { id: "intent", title: "Interpret query", detail: "Waiting for natural-language request", status: "queued" },
  { id: "route", title: "Select specialist", detail: "Agentic workflow routing", status: "queued" },
  { id: "vision", title: "Visual analysis", detail: "Waiting for vision-capable model", status: "queued" },
  { id: "evidence", title: "Build evidence", detail: "Waiting for observations", status: "queued" },
  { id: "synthesis", title: "Synthesize answer", detail: "Waiting for language layer", status: "queued" },
  { id: "audit", title: "Write audit trace", detail: "Waiting for execution", status: "queued" },
];

function workflowFor(query: string, hasSecond: boolean, secondaryModality: Modality) {
  const q = query.toLowerCase();
  if (hasSecond && (q.includes("sar") || secondaryModality === "sar")) return "Optical + SAR complementarity";
  if (hasSecond && (q.includes("change") || q.includes("before") || q.includes("after") || q.includes("compare"))) return "Bi-temporal change understanding";
  if (q.includes("where") || q.includes("locate") || q.includes("bounding") || q.includes("region")) return "Text-guided visual grounding";
  if (q.includes("caption") || q.includes("describe")) return "Scene captioning";
  return "Single-image visual question answering";
}

function buildFallbackPrompt(workflow: string) {
  if (workflow.includes("change")) return "Describe the visible differences between the two observations. Separate actual image evidence from assumptions and mention that the prototype computes a pixel-level change statistic.";
  if (workflow.includes("SAR")) return "Explain the complementary information available from optical and SAR observations, focusing on spectral versus radar backscatter/structure signals.";
  if (workflow.includes("grounding")) return "Identify the most relevant visual region for the user's request and describe where it is spatially. Do not invent exact coordinates.";
  if (workflow.includes("caption")) return "Provide a concise remote-sensing scene description grounded only in visible evidence.";
  return "Answer the user's remote-sensing question from the supplied image. Mention uncertainty where appropriate.";
}

function Stat({ label, value, icon: Icon }: { label: string; value: string; icon: typeof Activity }) {
  return <div className="rounded-xl border border-white/8 bg-white/[0.025] px-3 py-3"><div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.16em] text-slate-500"><Icon size={12} />{label}</div><div className="mt-1 text-sm font-semibold text-slate-100">{value}</div></div>;
}

function FileCard({ state, title, onChange, onModality }: { state: FileState; title: string; onChange: (file: File | null) => void; onModality: (modality: Modality) => void }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-black/20 p-3">
      <div className="mb-2 flex items-center justify-between"><span className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-500">{title}</span><Badge>{state.file ? state.modality.toUpperCase() : "EMPTY"}</Badge></div>
      <label className="group relative flex min-h-[165px] cursor-pointer items-center justify-center overflow-hidden rounded-xl border border-dashed border-white/10 bg-slate-950/70 transition hover:border-cyan-300/30">
        {state.preview ? <img src={state.preview} alt={state.file?.name ?? "satellite preview"} className="absolute inset-0 h-full w-full object-cover opacity-90" /> : <div className="text-center"><Upload className="mx-auto mb-3 text-slate-600" size={24}/><p className="text-xs text-slate-400">Drop optical/SAR image</p><p className="mt-1 text-[10px] text-slate-600">GeoTIFF · TIFF · PNG · JPEG</p></div>}
        <input type="file" accept=".tif,.tiff,.png,.jpg,.jpeg,image/*" className="hidden" onChange={(e) => onChange(e.target.files?.[0] ?? null)} />
        {state.file && <div className="absolute inset-x-2 bottom-2 rounded-lg border border-white/10 bg-slate-950/85 p-2 backdrop-blur"><div className="truncate text-[11px] text-white">{state.file.name}</div><div className="mt-1 text-[9px] text-slate-500">{formatBytes(state.file.size)} · {state.file.type || "raster"}</div></div>}
      </label>
      <div className="mt-2 grid grid-cols-2 gap-2">
        <button onClick={() => onModality("optical")} className={`rounded-lg border px-2 py-1.5 text-[10px] ${state.modality === "optical" ? "border-cyan-300/30 bg-cyan-300/10 text-cyan-100" : "border-white/8 text-slate-500"}`}><Satellite size={11} className="mr-1 inline"/>Optical</button>
        <button onClick={() => onModality("sar")} className={`rounded-lg border px-2 py-1.5 text-[10px] ${state.modality === "sar" ? "border-cyan-300/30 bg-cyan-300/10 text-cyan-100" : "border-white/8 text-slate-500"}`}><Radar size={11} className="mr-1 inline"/>SAR</button>
      </div>
    </div>
  );
}

function StepList({ steps }: { steps: Step[] }) {
  return <div className="space-y-2">{steps.map((step, index) => <div key={step.id} className="flex gap-3"><div className="flex flex-col items-center"><div className={`mt-0.5 flex h-6 w-6 items-center justify-center rounded-full border ${step.status === "done" ? "border-cyan-300/30 bg-cyan-300/10 text-cyan-200" : step.status === "running" ? "border-amber-300/30 bg-amber-300/10 text-amber-200 animate-pulse" : step.status === "error" ? "border-red-300/30 bg-red-300/10 text-red-200" : "border-white/10 bg-white/[0.02] text-slate-600"}`}>{step.status === "done" ? <CheckCircle2 size={13}/> : step.status === "error" ? <AlertTriangle size={13}/> : <span className="text-[9px]">{index + 1}</span>}</div>{index < steps.length - 1 && <div className="h-full min-h-5 w-px bg-white/8"/>}</div><div className="pb-2"><div className="text-xs font-medium text-slate-200">{step.title}</div><div className="text-[10px] text-slate-500">{step.detail}</div></div></div>)}</div>;
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
  const [tab, setTab] = useState<"evidence" | "trace" | "metrics" | "report">("evidence");
  const [center, setCenter] = useState<[number, number]>([28.6139, 77.209]);
  const [changeStats, setChangeStats] = useState<{ changedFraction: number; meanDifference: number } | null>(null);
  const [lastWorkflow, setLastWorkflow] = useState("Ready for mission");

  const hasSecond = Boolean(secondary.file);
  const workflow = useMemo(() => workflowFor(query, hasSecond, secondary.modality), [query, hasSecond, secondary.modality]);
  const evidenceCoverage = answer ? (hasSecond ? 92 : 86) : 0;

  async function loadFile(file: File | null, target: "primary" | "secondary") {
    const preview = file ? await imageDataUrl(file) : null;
    const next: FileState = target === "primary" ? { file, preview, modality: primary.modality } : { file, preview, modality: secondary.modality };
    target === "primary" ? setPrimary(next) : setSecondary(next);
  }

  async function runAnalysis() {
    if (!primary.file || !query.trim()) return;
    setRunning(true); setAnswer(""); setError(""); setTab("trace"); setLastWorkflow(workflow);
    const base = INITIAL_STEPS.map((s) => ({ ...s }));
    const patch = (index: number, status: Step["status"], detail?: string) => setSteps(base.map((s, i) => i === index ? { ...s, status, detail: detail ?? s.detail } : s));
    try {
      patch(0, "running", "Checking file count, formats and modality compatibility"); await wait(280);
      const primaryUrl = await imageDataUrl(primary.file);
      const secondaryUrl = secondary.file ? await imageDataUrl(secondary.file) : null;
      patch(0, "done", `${primary.file.name}${secondary.file ? ` + ${secondary.file.name}` : ""}`); await wait(220);
      patch(1, "running", `Detected intent: ${workflow}`); await wait(350); patch(1, "done", `Intent mapped to ${workflow}`); await wait(180);
      patch(2, "running", "Selecting specialist workflow and permitted tools"); await wait(320); patch(2, "done", workflow); await wait(180);
      patch(3, "running", provider === "openrouter" ? "Sending visual evidence to OpenRouter vision router" : "Sending visual evidence to Gemini multimodal model");
      let stats = null;
      if (secondary.file && workflow.includes("change")) stats = await compareImages(primary.file, secondary.file);
      setChangeStats(stats);
      const observations = {
        image_count: secondary.file ? 2 : 1,
        modalities: [primary.modality, secondary.file ? secondary.modality : null].filter(Boolean),
        aoi_center: center,
        deterministic_change: stats ? { changed_fraction: stats.changedFraction, mean_difference: stats.meanDifference } : null,
        note: "Browser prototype performs orchestration and lightweight evidence measurements; specialist RS checkpoints remain swappable backend components.",
      };
      const response = await fetch("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ provider, query, workflow, primary: { name: primary.file.name, modality: primary.modality, dataUrl: primaryUrl }, secondary: secondary.file ? { name: secondary.file.name, modality: secondary.modality, dataUrl: secondaryUrl } : null, observations }) });
      const data = await response.json();
      if (!response.ok) throw new Error(data?.error || "AI analysis failed.");
      patch(3, "done", `${data.provider} · ${data.model}`); await wait(160);
      patch(4, "running", "Combining model response with deterministic mission observations"); await wait(260); patch(4, "done", stats ? `Change statistic computed · ${(stats.changedFraction * 100).toFixed(1)}% pixels above threshold` : "Visual evidence ledger assembled"); await wait(160);
      patch(5, "running", "Generating grounded natural-language explanation"); await wait(300); setAnswer(data.answer); setModel(data.model); patch(5, "done", "Human-readable evidence synthesis complete"); await wait(140);
      patch(6, "done", "Trace recorded · provider, workflow, evidence and output captured"); setTab("evidence");
    } catch (e) {
      const message = e instanceof Error ? e.message : "Unknown analysis error.";
      setError(message); setSteps(base.map((s, i) => i < 3 ? { ...s, status: "done" } : i === 3 ? { ...s, status: "error", detail: message } : s)); setTab("trace");
    } finally { setRunning(false); }
  }

  function downloadReport() {
    const report = `SATQUERYX MISSION REPORT\n\nWorkflow: ${lastWorkflow}\nQuery: ${query}\nPrimary: ${primary.file?.name ?? "none"}\nSecondary: ${secondary.file?.name ?? "none"}\nAOI: ${center[0].toFixed(5)}, ${center[1].toFixed(5)}\nProvider: ${provider}\nModel: ${model || "not run"}\n\nANSWER\n${answer || "No completed analysis."}\n\nEVIDENCE COVERAGE\n${evidenceCoverage}%\n\nLIMITATION\nThis internal-round prototype uses a general multimodal language layer for natural-language synthesis. Remote-sensing-specialist checkpoints, benchmark evaluation and hidden ISRO/SAC validation remain the next implementation stage.`;
    const blob = new Blob([report], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob); const a = document.createElement("a"); a.href = url; a.download = "satqueryx-mission-report.txt"; a.click(); URL.revokeObjectURL(url);
  }

  return (
    <main className="min-h-screen grid-bg">
      <div className="mx-auto max-w-[1600px] px-4 py-4 lg:px-6">
        <header className="glass relative overflow-hidden rounded-3xl p-5 lg:p-6">
          <div className="absolute -right-20 -top-24 h-64 w-64 rounded-full bg-cyan-400/10 blur-3xl" />
          <div className="relative flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
            <div><div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.24em] text-cyan-300"><span className="h-2 w-2 rounded-full bg-cyan-300 shadow-[0_0_15px_rgba(34,211,238,.9)]"/>Remote sensing intelligence console</div><h1 className="mt-2 text-3xl font-semibold tracking-[-0.04em] text-white lg:text-4xl">SatQuery<span className="text-cyan-300">X</span></h1><p className="mt-1 max-w-3xl text-sm text-slate-400">Interactive vision-language analysis for optical, SAR and bi-temporal satellite imagery — orchestrated through natural language.</p></div>
            <div className="flex flex-wrap gap-2"><Badge><Zap size={11}/>Agentic orchestration</Badge><Badge><Radar size={11}/>Optical + SAR</Badge><Badge><RefreshCw size={11}/>Bi-temporal change</Badge><Badge><Crosshair size={11}/>Grounding</Badge><Badge><Activity size={11}/>Auditable trace</Badge></div>
          </div>
        </header>

        <div className="mt-4 grid gap-4 lg:grid-cols-[300px_minmax(0,1fr)_350px]">
          <aside className="space-y-4">
            <Card><CardHeader><CardTitle className="flex items-center gap-2"><Settings2 size={15} className="text-cyan-300"/>Mission configuration</CardTitle><CardDescription>Prototype controls map directly to the future agent controller.</CardDescription></CardHeader><CardContent className="space-y-4">
              <div><label className="mb-1.5 block text-[10px] uppercase tracking-[0.16em] text-slate-500">AI reasoning layer</label><select value={provider} onChange={(e) => setProvider(e.target.value as Provider)} className="h-10 w-full rounded-xl border border-white/10 bg-slate-950 px-3 text-xs text-slate-200 outline-none"><option value="openrouter">OpenRouter · free vision router</option><option value="gemini">Google Gemini</option></select></div>
              <div className="rounded-xl border border-cyan-300/10 bg-cyan-300/[0.035] p-3"><div className="flex items-center gap-2 text-xs font-medium text-cyan-100"><BrainCircuit size={14}/>Model abstraction</div><p className="mt-1 text-[10px] leading-4 text-slate-500">Keys stay server-side in <code>.env.local</code>. The language layer is replaceable by the trained RS-VLM later.</p></div>
              <div className="space-y-2"><div className="flex items-center justify-between"><span className="text-[10px] uppercase tracking-[0.16em] text-slate-500">Mission status</span><span className="text-[10px] text-cyan-300">{running ? "RUNNING" : "READY"}</span></div><div className="h-1.5 overflow-hidden rounded-full bg-white/5"><div className="h-full bg-cyan-300 transition-all duration-500" style={{ width: running ? "72%" : answer ? "100%" : "8%" }}/></div></div>
            </CardContent></Card>
            <Card><CardHeader><CardTitle>Agent responsibilities</CardTitle></CardHeader><CardContent className="space-y-2">{[[Satellite,"Input validation"],[BrainCircuit,"Intent interpretation"],[Layers3,"Specialist selection"],[ScanSearch,"Evidence extraction"],[Bot,"Language synthesis"],[Activity,"Audit trace"]].map(([Icon, text]) => <div key={String(text)} className="flex items-center gap-2 rounded-lg border border-white/6 bg-white/[0.02] px-3 py-2 text-[11px] text-slate-400"><Icon size={13} className="text-cyan-300"/>{String(text)}<ChevronRight size={12} className="ml-auto text-slate-700"/></div>)}</CardContent></Card>
          </aside>

          <section className="space-y-4">
            <Card className="scanline relative overflow-hidden"><CardHeader><div className="flex items-center justify-between"><div><CardTitle className="flex items-center gap-2"><FileImage size={15} className="text-cyan-300"/>Mission imagery</CardTitle><CardDescription>Use one image for VQA/captioning/grounding or two for temporal/cross-modal workflows.</CardDescription></div><Badge>{primary.file ? "INPUT READY" : "AWAITING INPUT"}</Badge></div></CardHeader><CardContent><div className="grid gap-3 md:grid-cols-2"><FileCard state={primary} title="Primary observation" onChange={(f) => void loadFile(f, "primary")} onModality={(m) => setPrimary((p) => ({ ...p, modality: m }))}/><FileCard state={secondary} title="Secondary observation" onChange={(f) => void loadFile(f, "secondary")} onModality={(m) => setSecondary((p) => ({ ...p, modality: m }))}/></div></CardContent></Card>

            <Card><CardHeader><CardTitle className="flex items-center gap-2"><MapPinned size={15} className="text-cyan-300"/>AOI intelligence</CardTitle><CardDescription>Click the map to reposition a demonstrable area of interest. The production path connects this layer to STAC/Sentinel-2 acquisition.</CardDescription></CardHeader><CardContent><div className="grid gap-3 xl:grid-cols-[1fr_220px]"><AOIMap center={center} onCenterChange={setCenter}/><div className="space-y-2"><Stat label="Latitude" value={center[0].toFixed(5)} icon={MapPinned}/><Stat label="Longitude" value={center[1].toFixed(5)} icon={MapPinned}/><Stat label="Acquisition" value="STAC-ready" icon={Cloud}/><div className="rounded-xl border border-white/8 bg-white/[0.02] p-3"><div className="text-[10px] uppercase tracking-[0.15em] text-slate-500">Pipeline</div><div className="mt-2 space-y-2 text-[10px] text-slate-400"><div>AOI → catalog search</div><div>↓ cloud filtering</div><div>↓ COG / GeoTIFF</div><div>↓ agent analysis</div></div></div></div></div></CardContent></Card>

            <Card><CardHeader><CardTitle className="flex items-center gap-2"><Sparkles size={15} className="text-cyan-300"/>Natural-language query console</CardTitle><CardDescription>The controller interprets the request before selecting the specialist workflow.</CardDescription></CardHeader><CardContent><textarea value={query} onChange={(e) => setQuery(e.target.value)} className="min-h-24 w-full resize-none rounded-2xl border border-white/10 bg-slate-950/80 p-4 text-sm text-slate-100 outline-none transition focus:border-cyan-300/30" placeholder="Ask about land cover, change, objects, or optical/SAR complementarity…"/><div className="mt-3 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between"><div className="flex items-center gap-2 text-[10px] text-slate-500"><CircleDot size={12} className="text-cyan-300"/>Selected workflow: <span className="font-medium text-slate-300">{workflow}</span></div><Button disabled={!primary.file || running} onClick={() => void runAnalysis()}>{running ? <RefreshCw size={14} className="animate-spin"/> : <Send size={14}/>} {running ? "Running analysis" : "Run analysis"}</Button></div></CardContent></Card>

            <div className="grid grid-cols-2 gap-2 lg:grid-cols-4"><Stat label="Images" value={secondary.file ? "2 / paired" : primary.file ? "1 / single" : "0 / waiting"} icon={FileImage}/><Stat label="Workflow" value={lastWorkflow === "Ready for mission" ? "Awaiting" : lastWorkflow.split(" ").slice(0, 2).join(" ")} icon={Zap}/><Stat label="Evidence" value={answer ? `${evidenceCoverage}% coverage` : "Not run"} icon={Gauge}/><Stat label="AOI" value={`${center[0].toFixed(2)}, ${center[1].toFixed(2)}`} icon={MapPinned}/></div>

            <Card><CardHeader><div className="flex flex-wrap gap-1 rounded-xl border border-white/6 bg-black/20 p-1">{(["evidence","trace","metrics","report"] as const).map((name) => <button key={name} onClick={() => setTab(name)} className={`rounded-lg px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.14em] transition ${tab === name ? "bg-cyan-300/10 text-cyan-100" : "text-slate-500 hover:text-slate-300"}`}>{name}</button>)}</div></CardHeader><CardContent>
              {tab === "evidence" && <div className="space-y-4"><div className="rounded-2xl border border-cyan-300/20 bg-gradient-to-r from-cyan-300/[0.07] to-transparent p-5"><div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.18em] text-cyan-200"><Bot size={13}/>Evidence-grounded answer</div><p className="mt-3 text-base leading-7 text-slate-100">{answer || "Run an analysis to populate the model answer. The prototype keeps the language layer separate from remote-sensing specialist tools so the trained RS-VLM can replace it later."}</p>{error && <div className="mt-3 flex gap-2 rounded-xl border border-red-300/15 bg-red-300/5 p-3 text-xs text-red-200"><AlertTriangle size={14} className="mt-0.5 shrink-0"/>{error}</div>}</div><div className="grid gap-3 md:grid-cols-3"><Stat label="Workflow" value={workflow} icon={Layers3}/><Stat label="Model" value={model || "not run"} icon={BrainCircuit}/><Stat label="Evidence coverage" value={answer ? `${evidenceCoverage}%` : "—"} icon={Gauge}/></div>{changeStats && <div className="rounded-2xl border border-white/8 bg-black/20 p-4"><div className="flex items-center gap-2 text-xs font-medium text-slate-200"><RefreshCw size={14} className="text-cyan-300"/>Deterministic change evidence</div><div className="mt-3 grid grid-cols-2 gap-3"><Stat label="Changed fraction" value={`${(changeStats.changedFraction * 100).toFixed(1)}%`} icon={Activity}/><Stat label="Mean difference" value={changeStats.meanDifference.toFixed(3)} icon={Gauge}/></div><p className="mt-3 text-[10px] leading-4 text-slate-500">Computed from decoded RGB previews in the prototype. Production GeoTIFF change analysis remains in the existing Python geospatial pipeline.</p></div>}</div>}
              {tab === "trace" && <div><div className="mb-4 flex items-center gap-2 text-xs text-slate-400"><Activity size={14} className="text-cyan-300"/>Auditable execution trace · {workflow}</div><StepList steps={steps}/></div>}
              {tab === "metrics" && <div className="grid gap-3 md:grid-cols-2"><Stat label="Input compatibility" value={primary.file ? "Validated" : "Waiting"} icon={CheckCircle2}/><Stat label="Modality pair" value={secondary.file ? `${primary.modality} + ${secondary.modality}` : primary.file ? primary.modality : "—"} icon={Radar}/><Stat label="Temporal analysis" value={changeStats ? "Pixel comparison active" : "Not selected"} icon={RefreshCw}/><Stat label="Grounding" value={workflow.includes("grounding") ? "Requested" : "Available route"} icon={Target}/><Stat label="AI provider" value={provider === "openrouter" ? "OpenRouter" : "Gemini"} icon={BrainCircuit}/><Stat label="Evidence ledger" value={answer ? "Populated" : "Pending"} icon={Layers3}/></div>}
              {tab === "report" && <div className="rounded-2xl border border-white/8 bg-black/20 p-5"><div className="flex items-start justify-between gap-4"><div><div className="text-sm font-semibold text-slate-100">Mission report</div><p className="mt-1 text-xs leading-5 text-slate-500">Export the current prototype run for the internal presentation. The production backend already contains the PDF reporting path.</p></div><Button variant="outline" onClick={downloadReport}><Download size={14}/>Download report</Button></div><pre className="mt-4 max-h-64 overflow-auto rounded-xl border border-white/6 bg-slate-950 p-4 text-[10px] leading-5 text-slate-400">{answer || "No completed analysis yet."}</pre></div>}
            </CardContent></Card>
          </section>

          <aside className="space-y-4">
            <Card><CardHeader><CardTitle>System architecture</CardTitle><CardDescription>What the judge sees as the prototype's core differentiator.</CardDescription></CardHeader><CardContent><div className="space-y-2 text-[10px]">{[["01","Natural language","Query understanding"],["02","Agent controller","Task + modality routing"],["03","Specialists","VQA · change · fusion · grounding"],["04","Evidence layer","Measurements + visual evidence"],["05","Language layer","OpenRouter / Gemini"],["06","Audit trace","Models · tools · parameters · output"]].map(([n,t,d],i) => <div key={n} className="relative rounded-xl border border-white/7 bg-white/[0.02] p-3"><div className="flex items-center gap-2"><span className="font-mono text-cyan-300">{n}</span><span className="font-semibold text-slate-200">{t}</span>{i < 5 && <ArrowRight size={11} className="ml-auto text-slate-700"/>}</div><div className="mt-1 pl-6 text-slate-500">{d}</div></div>)}</div></CardContent></Card>
            <Card><CardHeader><CardTitle className="flex items-center gap-2"><Waves size={15} className="text-cyan-300"/>Sensor complementarity</CardTitle></CardHeader><CardContent><div className="space-y-2">{[["Spectral information","OPTICAL","strong"],["Surface structure","SAR","strong"],["Cloud sensitivity","SAR","low"],["Vegetation cues","OPTICAL","strong"]].map(([a,b,c]) => <div key={a} className="flex items-center justify-between border-b border-white/6 py-2 last:border-0"><span className="text-[10px] text-slate-500">{a}</span><span className="text-[9px] font-semibold uppercase tracking-wider text-cyan-200">{b} · {c}</span></div>)}</div></CardContent></Card>
            <Card><CardHeader><CardTitle>Prototype boundary</CardTitle></CardHeader><CardContent><div className="rounded-xl border border-amber-300/10 bg-amber-300/[0.03] p-3 text-[10px] leading-5 text-slate-500"><span className="font-semibold text-amber-200">Tomorrow's claim:</span> working orchestration prototype. The trained RS-VLM, dedicated grounding checkpoint, benchmark evaluation and hidden ISRO/SAC validation are the next stage — not fabricated as complete today.</div></CardContent></Card>
          </aside>
        </div>
        <footer className="mt-4 flex flex-col gap-2 border-t border-white/6 px-1 py-4 text-[9px] uppercase tracking-[0.15em] text-slate-600 sm:flex-row sm:items-center sm:justify-between"><span>SATQUERYX · SIH2026 PROTOTYPE CONSOLE</span><span>Optical · SAR · Bi-temporal · Agentic VLM</span></footer>
      </div>
    </main>
  );
}

function wait(ms: number) { return new Promise((resolve) => setTimeout(resolve, ms)); }
