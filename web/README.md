# SatQueryX React Intelligence Console

Judge-facing Next.js console for SatQueryX. This UI sits beside the existing Python/Streamlit research and geospatial backend; it is intentionally built as the product layer for the SIH2026 demonstration flow.

## Run from the repository root

The `web` directory is inside the SatQueryX repository. In GitHub Codespaces, start from the directory shown by the terminal prompt (normally `/workspaces/SatQueryX`):

```bash
cd /workspaces/SatQueryX/web
npm install
cp .env.example .env.local
npm run dev
```

If your terminal is already at the repository root, use:

```bash
cd web
npm install
cp .env.example .env.local
npm run dev
```

Do **not** run `cd SatQueryX/web` from a terminal that is already inside `/workspaces/SatQueryX`; that tries to enter a second nested `SatQueryX` directory and causes `No such file or directory`.

Open the forwarded port for `3000` in Codespaces.

## API keys

Put credentials in `web/.env.local`. Keep AI keys server-side; do not use `NEXT_PUBLIC_*` for secrets.

```env
AI_PROVIDER=openrouter
OPENROUTER_API_KEY=
OPENROUTER_MODEL=openrouter/free
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3-flash-preview
```

When a provider is not configured, SatQueryX reports that state and does not fabricate an answer.

## Product flow

Upload imagery → choose/confirm modality → trace or move the AOI → enter a natural-language request → run the agentic workflow → inspect the scene/evidence → open the audit trace → export the mission report.

The console also exposes the 3D Earth Explorer, temporal comparison, optical/SAR workflow, grounding overlay, contextual Copilot, and execution telemetry.

## Architecture boundary

The React layer does not replace the scientific backend. It is the judge-facing orchestration/product surface for the existing SatQueryX pipeline, including GeoTIFF validation, remote-sensing specialists, PaliGemma/QLoRA training, change/fusion tooling, prescribed benchmark evaluation, and the eventual hidden ISRO/SAC validation path.
