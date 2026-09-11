# SatQueryX React Intelligence Console

A judge-facing internal-round prototype UI for SatQueryX. The existing Python/Streamlit application remains the research/geospatial backend; this Next.js application provides a polished interactive console for the demo path.

## Stack

- Next.js App Router + React + TypeScript
- Tailwind CSS v4
- shadcn-compatible `components/ui` structure
- Lucide icons
- React Leaflet + OpenStreetMap for AOI visualization
- Server-side `/api/analyze` route for OpenRouter or Gemini multimodal inference

## Run

```bash
cd web
npm install
cp .env.example .env.local
npm run dev
```

Open `http://localhost:3000`.

## API keys

Put credentials in `web/.env.local`. **Do not use `NEXT_PUBLIC_*` for AI keys.**

```env
AI_PROVIDER=openrouter
OPENROUTER_API_KEY=
OPENROUTER_MODEL=openrouter/free
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3-flash-preview
```

The default OpenRouter route uses `openrouter/free`, which supports image input and text output. Gemini is available as an alternate provider. If the selected provider is not configured, the UI reports the missing credential instead of fabricating an answer.

## Demo flow

1. Upload one JPG/PNG satellite image and ask a VQA/caption question.
2. Run the analysis and open **Agent trace** while the orchestration steps animate.
3. Upload a second image, mark it SAR or optical, and ask a change or complementarity question.
4. Click the AOI map to move the area of interest and explain the STAC → GeoTIFF acquisition path.
5. Open **Metrics** to show evidence coverage and deterministic change statistics.
6. Open **Report** and export the current mission summary.

## Prototype boundary

This web layer intentionally uses a general multimodal model for natural-language synthesis. It does not claim that the general model is the final remote-sensing specialist. The production roadmap is to connect the existing Python geospatial tools, trained PaliGemma/QLoRA RS-VLM adapter, dedicated grounding/change checkpoints, BigEarthNet training, prescribed benchmarks, and hidden ISRO/SAC evaluation behind the same orchestration contract.
