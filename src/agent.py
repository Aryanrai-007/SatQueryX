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
    workflow: str = "single_vqa"
    datasets: list[str] = field(default_factory=list)
    steps: list[ExecutionStep] = field(default_factory=list)


class AgenticOrchestrator:
    """Observable, deterministic SIH workflow router.

    The planner does not expose private chain-of-thought. It returns only the
    selected workflow, specialist tools, required inputs and public dataset
    provenance that can be shown in the execution trace.
    """

    def classify_intent(self, query: str) -> list[str]:
        q = query.lower()
        intents: list[str] = []
        if any(x in q for x in ("caption", "describe", "scene description", "land-cover and major objects")):
            intents.append("captioning")
        if any(x in q for x in ("highlight", "ground", "localize", "where is", "where are", "region")):
            intents.append("text_grounding")
        if any(x in q for x in ("change", "changed", "compare", "comparison", "before", "after", "difference", "damage", "temporal", "increased", "decreased")):
            intents.append("change_understanding")
        if ("optical" in q and "sar" in q) or any(x in q for x in ("cross-modal", "cross modal", "radar and optical", "use both sensors")):
            intents.append("optical_sar")
        if any(x in q for x in ("ndvi", "vegetation", "crop", "greenness", "health")):
            intents.append("ndvi")
        if any(x in q for x in ("sar", "radar", "backscatter", "microwave")):
            intents.append("sar")
        if any(x in q for x in ("detect", "find", "identify", "object", "vehicle", "building", "road", "ship")):
            intents.append("visual_grounding")
        if not intents:
            intents.append("visual_question_answering")
        return list(dict.fromkeys(intents))

    def plan(self, query: str, has_second_image: bool = False, modalities: tuple[str, ...] | None = None) -> AnalysisPlan:
        intents = self.classify_intent(query)
        q = query.lower()
        modalities = tuple(modalities or ())

        if "optical_sar" in intents:
            workflow = "optical_sar"
            tools = ["optical_sar_fusion", "multimodal_rs_vlm"]
            datasets = ["BigEarthNet.txt"]
        elif "change_understanding" in intents:
            workflow = "bi_temporal_change"
            tools = ["change_detection", "change_vqa"]
            datasets = ["CDVQA"]
        elif "text_grounding" in intents:
            workflow = "text_grounding"
            tools = ["rs_grounding"]
            datasets = ["BigEarthNet.txt", "VRSBench"]
        elif "captioning" in intents:
            workflow = "single_captioning"
            tools = ["rs_captioning"]
            datasets = ["BigEarthNet.txt", "VRSBench"]
        else:
            workflow = "single_vqa"
            tools = ["rs_vqa"]
            datasets = ["BigEarthNet.txt", "RSVQA"]

        # Deterministic auxiliary analysis can be composed with the primary workflow.
        if "ndvi" in intents:
            tools.append("ndvi")
        if "sar" in intents and "sar_statistics" not in tools:
            tools.append("sar_statistics")
        if "visual_grounding" in intents and "rs_grounding" not in tools:
            tools.append("visual_grounding")

        # If the user explicitly asks to use both images but did not mention a
        # modality, temporal reasoning is the safe default; validation below still
        # checks compatibility before execution.
        if has_second_image and workflow == "single_vqa" and any(x in q for x in ("both images", "two images", "two dates")):
            workflow = "bi_temporal_change"
            tools = ["change_detection", "change_vqa"]
            datasets = ["CDVQA"]

        return AnalysisPlan(intents=intents, tools=list(dict.fromkeys(tools)), workflow=workflow, datasets=datasets)

    @staticmethod
    def validate_required_inputs(plan: AnalysisPlan, has_image: bool, has_second_image: bool, modalities: tuple[str, ...] | None = None) -> list[str]:
        errors: list[str] = []
        modalities = tuple(modalities or ())
        if not has_image:
            errors.append("At least one image is required.")
        if plan.workflow in {"bi_temporal_change", "optical_sar"} and not has_second_image:
            errors.append(f"Workflow '{plan.workflow}' requires two spatially corresponding images.")
        if plan.workflow == "optical_sar" and modalities and not ({"optical", "sar"} <= set(modalities)):
            errors.append("Optical/SAR workflow requires one optical/multispectral input and one SAR input.")
        if plan.workflow == "bi_temporal_change" and len(modalities) >= 2 and modalities[0] != modalities[1]:
            errors.append("Bi-temporal change analysis expects comparable observations; use the same modality or a dedicated cross-modal workflow.")
        return errors
