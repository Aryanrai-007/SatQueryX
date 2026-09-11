import { NextResponse } from "next/server";

export const runtime = "nodejs";

const systemPrompt = `You are the language/evidence synthesis layer inside SatQueryX, an interactive remote-sensing analysis prototype. You are NOT the remote-sensing specialist itself. Treat the supplied image observations and workflow as evidence. Never invent sensor metadata, coordinates, labels, measurements, benchmark scores, or detections that are not provided. Explain uncertainty explicitly. If an image is only a generic RGB view, say so. For bi-temporal requests, distinguish visual observations from deterministic change metrics. For optical/SAR requests, explain complementary physical signals without pretending the image is SAR unless the user marked it as SAR. Keep the answer concise but technically useful for a judge-facing prototype.`;

type RequestBody = {
  provider?: "openrouter" | "gemini";
  model?: string;
  query: string;
  workflow: string;
  primary?: { name: string; modality: string; dataUrl?: string | null };
  secondary?: { name: string; modality: string; dataUrl?: string | null } | null;
  observations?: Record<string, unknown>;
};

function imagePart(dataUrl: string | null | undefined) {
  if (!dataUrl) return null;
  return { type: "image_url", image_url: { url: dataUrl } };
}

async function callOpenRouter(body: RequestBody) {
  const key = process.env.OPENROUTER_API_KEY;
  if (!key) return { configured: false, message: "OPENROUTER_API_KEY is not configured on the server." };
  const model = body.model || process.env.OPENROUTER_MODEL || "openrouter/free";
  const content = [
    { type: "text", text: `${systemPrompt}\n\nWORKFLOW: ${body.workflow}\nUSER QUERY: ${body.query}\nOBSERVATIONS: ${JSON.stringify(body.observations ?? {})}\nPRIMARY: ${body.primary?.name ?? "none"} (${body.primary?.modality ?? "unknown"})\nSECONDARY: ${body.secondary?.name ?? "none"} (${body.secondary?.modality ?? "unknown"})` },
    imagePart(body.primary?.dataUrl),
    imagePart(body.secondary?.dataUrl),
  ].filter(Boolean);
  const response = await fetch("https://openrouter.ai/api/v1/chat/completions", { method: "POST", headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json", "X-Title": "SatQueryX Remote Sensing Intelligence Console" }, body: JSON.stringify({ model, messages: [{ role: "user", content }], temperature: 0.15, max_tokens: 420 }) });
  const data = await response.json();
  if (!response.ok) throw new Error(data?.error?.message || `OpenRouter request failed (${response.status}).`);
  return { configured: true, answer: data?.choices?.[0]?.message?.content ?? "The model returned no textual answer.", model: data?.model ?? model, provider: "OpenRouter" };
}

async function callGemini(body: RequestBody) {
  const key = process.env.GEMINI_API_KEY;
  if (!key) return { configured: false, message: "GEMINI_API_KEY is not configured on the server." };
  const model = body.model || process.env.GEMINI_MODEL || "gemini-3-flash-preview";
  const parts: Array<Record<string, unknown>> = [{ text: `${systemPrompt}\n\nWORKFLOW: ${body.workflow}\nUSER QUERY: ${body.query}\nOBSERVATIONS: ${JSON.stringify(body.observations ?? {})}\nPRIMARY: ${body.primary?.name ?? "none"} (${body.primary?.modality ?? "unknown"})\nSECONDARY: ${body.secondary?.name ?? "none"} (${body.secondary?.modality ?? "unknown"})` }];
  for (const item of [body.primary, body.secondary]) {
    const dataUrl = item?.dataUrl;
    if (!dataUrl) continue;
    const match = dataUrl.match(/^data:([^;]+);base64,(.+)$/);
    if (match) parts.push({ inlineData: { mimeType: match[1], data: match[2] } });
  }
  const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent?key=${encodeURIComponent(key)}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ contents: [{ role: "user", parts }] }) });
  const data = await response.json();
  if (!response.ok) throw new Error(data?.error?.message || `Gemini request failed (${response.status}).`);
  const answer = data?.candidates?.[0]?.content?.parts?.map((part: { text?: string }) => part.text).filter(Boolean).join("\n") || "The model returned no textual answer.";
  return { configured: true, answer, model, provider: "Gemini" };
}

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as RequestBody;
    if (!body.query?.trim()) return NextResponse.json({ error: "A natural-language query is required." }, { status: 400 });
    const provider = body.provider || (process.env.AI_PROVIDER === "gemini" ? "gemini" : "openrouter");
    const result = provider === "gemini" ? await callGemini(body) : await callOpenRouter(body);
    return NextResponse.json({ ...result, timestamp: new Date().toISOString() });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown analysis error.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
