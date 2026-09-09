from src.site_analysis import WORLD_COVER_CLASSES, _bbox_area_m2


def test_bbox_area_is_positive_for_small_aoi():
    area = _bbox_area_m2((77.217, 28.678, 77.221, 28.682))
    assert 100_000 < area < 250_000


def test_worldcover_classes_include_requested_context():
    assert WORLD_COVER_CLASSES[30] == "Grassland"
    assert WORLD_COVER_CLASSES[80] == "Permanent water"
    assert WORLD_COVER_CLASSES[50] == "Built-up"
