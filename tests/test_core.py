import numpy as np

from src.agent import AgenticOrchestrator
from src.tools.change_detection import detect_change
from src.tools.optical_sar_fusion import fuse_optical_sar


def test_change_detection_identical_arrays():
    a = np.ones((32, 32), dtype=np.float32)
    r = detect_change(a, a)
    assert r.changed_fraction == 0.0
    assert r.ssim == 1.0


def test_fusion_shape():
    a = np.arange(100, dtype=np.float32).reshape(10, 10)
    b = np.flipud(a)
    r = fuse_optical_sar(a, b)
    assert r.fused.shape == a.shape


def test_agent_change_plan_requires_second_image():
    plan = AgenticOrchestrator().plan("compare before and after for change")
    assert "change_detection" not in plan.tools
    plan2 = AgenticOrchestrator().plan("compare before and after for change", has_second_image=True)
    assert "change_detection" in plan2.tools
