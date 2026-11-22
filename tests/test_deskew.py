import numpy as np
import pytest

import synth
from naturaw.deskew import (DeskewParams, deskew_image, estimate_tilt,
                            largest_inner_rect, largest_inner_rect_same_aspect,
                            rotate_image)


@pytest.fixture(scope="module")
def building():
    return synth.architecture(700, 1050)


@pytest.mark.parametrize("angle", [0.4, -1.1, 2.6, -5.0, 7.2])
def test_tilt_is_recovered(building, angle):
    tilted = rotate_image(building, angle, crop="aspect")
    est = estimate_tilt(tilted, DeskewParams())
    assert abs(est.angle + angle) < 0.25
    assert est.confidence > 0.3


def test_straight_image_is_left_alone(building):
    report = {}
    out = deskew_image(building, DeskewParams(), report=report)
    assert out.shape == building.shape
    assert "ignore" in report["deskew"]


def test_textured_scene_without_lines_is_not_rotated():
    tex = np.dstack([synth.texture(400, 600, s) for s in (1, 2, 3)]).astype(np.float32)
    report = {}
    out = deskew_image(np.clip(tex, 0.01, 1), DeskewParams(), report=report)
    assert out.shape == tex.shape


def test_rotation_is_applied_and_cropped(building):
    tilted = rotate_image(building, 3.0, crop="aspect")
    report = {}
    out = deskew_image(tilted, DeskewParams(), report=report)
    assert "redressee" in report["deskew"]
    assert out.shape[0] < tilted.shape[0] and out.shape[1] < tilted.shape[1]


@pytest.mark.parametrize("angle", [1.0, 4.0, 9.0])
def test_aspect_crop_keeps_proportions(angle):
    w, h = 1200.0, 800.0
    wr, hr = largest_inner_rect_same_aspect(w, h, angle)
    assert abs((wr / hr) - (w / h)) < 1e-6
    assert 0.5 < (wr * hr) / (w * h) < 1.0


def test_max_crop_has_larger_area_than_aspect_crop():
    w, h = 1200.0, 800.0
    a = np.prod(largest_inner_rect_same_aspect(w, h, 6.0))
    b = np.prod(largest_inner_rect(w, h, 6.0))
    assert b >= a


def test_zero_angle_keeps_full_frame():
    wr, hr = largest_inner_rect_same_aspect(1000.0, 500.0, 0.0)
    assert abs(wr - 1000.0) < 1e-6 and abs(hr - 500.0) < 1e-6


def test_forced_angle_is_used(building):
    report = {}
    out = deskew_image(building, DeskewParams(angle=-2.0), report=report)
    assert "impose" in report["deskew"]
    assert out.shape != building.shape
