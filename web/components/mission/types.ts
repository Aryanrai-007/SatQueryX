export type Modality = "optical" | "sar";
export type Provider = "openrouter" | "gemini";
export type StepStatus = "queued" | "running" | "done" | "error" | "waiting";
export type Step = { id: string; title: string; detail: string; status: StepStatus };
export type FileState = { file: File | null; preview: string | null; modality: Modality };
export type MissionResult = any;
