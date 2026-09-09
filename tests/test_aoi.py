from datetime import date

from src.aoi import bbox_center, geometry_bbox, select_items


def test_geometry_bbox_and_center():
    geometry = {
        "type": "Polygon",
        "coordinates": [[[77.0, 28.0], [77.2, 28.0], [77.2, 28.2], [77.0, 28.2], [77.0, 28.0]]],
    }
    bbox = geometry_bbox(geometry)
    assert bbox == (77.0, 28.0, 77.2, 28.2)
    assert bbox_center(bbox) == (28.1, 77.1)


def test_select_two_temporally_distinct_items():
    features = [
        {"id": "latest", "properties": {"datetime": "2026-09-01T10:00:00Z"}},
        {"id": "too_close", "properties": {"datetime": "2026-08-25T10:00:00Z"}},
        {"id": "previous", "properties": {"datetime": "2026-07-01T10:00:00Z"}},
    ]
    selected = select_items(features, count=2, min_gap_days=14)
    assert [item["id"] for item in selected] == ["latest", "previous"]
