from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExecutionStep:
    name: str
    status: str
    detail: str


@dataclass
class AnalysisPlan:
    intents: list[str]
    tools: list[str]
    steps: list[ExecutionStep] = field(default_factory=list)


class AgenticOrchestrator:
    """Deterministic planner that maps natural-language requests to real tools."""

    def classify_intent(self, query: str) -> list[str]:
        q = query.lower().strip()
        intents: list[str] = []
        if any(x in q for x in ("detect", "find", "identify", "object", "vehicle", "building", "road", "ship")):
            intents.append("visual_grounding")
        if any(x in q for x in ("change", "changed", "compare", "comparison", "compare the", "before", "after", "difference", "damage", "temporal")):
            intents.append("change_detection")
        if any(x in q for x in ("ndvi", "vegetation", "crop", "greenness", "health", "plant")):
            intents.append("ndvi")
        if any(x in q for x in ("sar", "radar", "backscatter", "microwave")):
            intents.append("sar")
        if any(x in q for x in ("optical", "multispectral", "rgb")):
            intents.append("optical")
        if any(x in q for x in ("fuse", "fusion", "combine optical", "combine sar")) or ("optical" in q and "sar" in q):
            intents.append("fusion")

        # "Analyze", "summarize", and similar broad requests should not collapse
        # to metadata-only VQA. The app has a real 4-band Sentinel-2 path, so the
        # planner requests NDVI as a concrete spectral result when appropriate.
        broad = any(x in q for x in ("analyze", "analyse", "summary", "summarize", "overview", "assess", "assessment", "describe this area"))
        if broad:
            if "ndvi" not in intents:
                intents.append("ndvi")
            intents.append("visual_question_answering")
        elif not intents:
            intents.append("visual_question_answering")
        return list(dict.fromkeys(intents))

    def plan(self, query: str, has_second_image: bool = False) -> AnalysisPlan:
        intents = self.classify_intent(query)
        tools: list[str] = []
        if "visual_grounding" in intents:
            tools.append("visual_grounding")
        if "change_detection" in intents and has_second_image:
            tools.append("change_detection")
        if "ndvi" in intents:
            tools.append("ndvi")
        if "fusion" in intents and has_second_image:
            tools.append("optical_sar_fusion")
        if "sar" in intents:
            tools.append("sar_statistics")
        if "visual_question_answering" in intents:
            tools.append("vlm_reasoning")
        return AnalysisPlan(intents=intents, tools=list(dict.fromkeys(tools)))

    @staticmethod
    def validate_required_inputs(plan: AnalysisPlan, has_image: bool, has_second_image: bool) -> list[str]:
        errors: list[str] = []
        if not has_image:
            errors.append("At least one image is required.")
        if "change_detection" in plan.intents and not has_second_image:
            errors.append("This query requests comparison/change analysis, which requires a second image.")
        if "optical_sar_fusion" in plan.tools and not has_second_image:
            errors.append("Optical/SAR fusion requires both optical and SAR inputs.")
        return errors
