import { NextResponse } from "next/server";

export const runtime = "nodejs";

const systemPrompt = `You are the language/evidence synthesis layer inside SatQueryX, an interactive remote-sensing analysis prototype. You are NOT the remote-sensing specialist itself. Treat the supplied image observations and workflow as evidence. Never invent sensor metadata, coordinates, labels, measurements, benchmark scores, or detections that are not provided. Explain uncertainty explicitly. If an image is only a generic RGB view, say so. For bi-temporal requests, distinguish visual observations from deterministic change metrics. For optical/SAR requests, explain complementary physical signals without pretending the image is SAR unless the user marked it as SAR. Keep the answer concise but technically useful for a judge-facing prototype.`;

type Provider = "openrouter" | "gemini";
type RequestBody = {
  provider?: Provider;
  model?: string;
  query: string;
  workflow: string;
  primary?: { name: string; modality: string; dataUrl?: string | null };
  secondary?: { name: string; modality: string; dataUrl?: string | null } | null;
  observations?: Record<string, unknown>;
};

function promptText(body: RequestBody) {
  return `${systemPrompt}\n\nWORKFLOW: ${body.workflow}\nUSER QUERY: ${body.query}\nOBSERVATIONS: ${JSON.stringify(body.observations ?? {})}\nPRIMARY: ${body.primary?.name ?? "none"} (${body.primary?.modality ?? "unknown"})\nSECONDARY: ${body.secondary?.name ?? "none"} (${body.secondary?.modality ?? "unknown"})`;
}

function imagePart(dataUrl: string | null | undefined) {
  if (!dataUrl) return null;
  return { type: "image_url", image_url: { url: dataUrl } };
}

async function callOpenRouter(body: RequestBody) {
  const key = process.env.OPENROUTER_API_KEY;
  if (!key) return { configured: false, missing: true, message: "OPENROUTER_API_KEY is not configured on the server." };
  const model = body.model || process.env.OPENROUTER_MODEL || "openrouter/free";
  const content = [
    { type: "text", text: promptText(body) },
    imagePart(body.primary?.dataUrl),
    imagePart(body.secondary?.dataUrl),
  ].filter(Boolean);
  const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${key}`,
      "Content-Type": "application/json",
      "X-Title": "SatQueryX Remote Sensing Intelligence Console",
    },
    body: JSON.stringify({
      model,
      messages: [{ role: "user", content }],
      temperature: 0.15,
      max_tokens: 520,
    }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data?.error?.message || `OpenRouter request failed (${response.status}).`);
  return {
    configured: true,
    answer: data?.choices?.[0]?.message?.content ?? "The model returned no textual answer.",
    model: data?.model ?? model,
    provider: "OpenRouter",
  };
}

async function callGemini(body: RequestBody) {
  const key = process.env.GEMINI_API_KEY;
  if (!key) return { configured: false, missing: true, message: "GEMINI_API_KEY is not configured on the server." };
  const model = body.model || process.env.GEMINI_MODEL || "gemini-3.8-flash";
  const parts: Array<Record<string, unknown>> = [{ text: promptText(body) }];
  for (const item of [body.primary, body.secondary]) {
    const dataUrl = item?.dataUrl;
    if (!dataUrl) continue;
    const match = dataUrl.match(/^data:([^;]+);base64,(.+)$/);
    if (match) parts.push({ inlineData: { mimeType: match[1], data: match[2] } });
  }
  const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent?key=${encodeURIComponent(key)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [{ role: "user", parts }],
      generationConfig: { maxOutputTokens: 520 },
    }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data?.error?.message || `Gemini request failed (${response.status}).`);
  const answer = data?.candidates?.[0]?.content?.parts?.map((part: { text?: string }) => part.text).filter(Boolean).join("\n") || "The model returned no textual answer.";
  return { configured: true, answer, model, provider: "Gemini" };
}

async function runProvider(provider: Provider, body: RequestBody) {
  return provider === "gemini" ? callGemini(body) : callOpenRouter(body);
}

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as RequestBody;
    if (!body.query?.trim()) return NextResponse.json({ error: "A natural-language query is required." }, { status: 400 });

    const requested = body.provider || (process.env.AI_PROVIDER === "openrouter" ? "openrouter" : "gemini");
    const primary = await runProvider(requested, body);
    if (primary.configured) return NextResponse.json({ ...primary, timestamp: new Date().toISOString() });

    const alternate: Provider = requested === "gemini" ? "openrouter" : "gemini";
    const fallback = await runProvider(alternate, body);
    if (fallback.configured) {
      return NextResponse.json({
        ...fallback,
        fallback: true,
        requestedProvider: requested,
        timestamp: new Date().toISOString(),
      });
    }

    return NextResponse.json({
      configured: false,
      message: `No AI provider is configured. Add ${requested === "gemini" ? "GEMINI_API_KEY" : "OPENROUTER_API_KEY"} or the alternate provider key in web/.env.local.`,
      providers: { gemini: Boolean(process.env.GEMINI_API_KEY), openrouter: Boolean(process.env.OPENROUTER_API_KEY) },
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown analysis error.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
