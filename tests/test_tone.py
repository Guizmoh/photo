import numpy as np
import pytest

import synth
from naturaw.colorspace import luminance
from naturaw.tone import ToneParams, auto_shadow_amount, tone_rescue


@pytest.fixture(scope="module")
def scene():
    return synth.architecture(400, 600)


def test_correctly_exposed_image_is_left_alone(scene):
    report = {}
    out = tone_rescue(scene, ToneParams(), report=report)
    assert report["tonalite"] == "inchangee"
    assert out is scene


@pytest.mark.parametrize("ev", [-1.0, -2.0, -3.0])
def test_underexposed_images_are_lifted(scene, ev):
    dark = np.clip(scene * 2.0 ** ev, 0, 1).astype(np.float32)
    assert auto_shadow_amount(dark) > 0.05
    out = tone_rescue(dark, ToneParams())
    assert np.median(luminance(out)) > np.median(luminance(dark)) * 1.2


def test_highlights_are_not_blown(scene):
    dark = np.clip(scene * 0.125, 0, 1).astype(np.float32)
    out = tone_rescue(dark, ToneParams())
    assert float(np.percentile(luminance(out), 99.9)) < 1.0


def test_colours_are_preserved(scene):
    dark = np.clip(scene * 0.25, 0, 1).astype(np.float32)
    out = tone_rescue(dark, ToneParams())
    lum_in = np.maximum(luminance(dark), 1e-5)
    lum_out = np.maximum(luminance(out), 1e-5)
    # le rapport entre canaux (donc la teinte) ne doit pas bouger
    ratio_in = dark[..., 0] / lum_in
    ratio_out = out[..., 0] / lum_out
    assert float(np.abs(ratio_in - ratio_out).max()) < 0.02


def test_no_halo_around_a_strong_edge():
    """Le gain doit suivre le contour, pas deborder autour."""
    img = np.full((200, 200, 3), 0.03, np.float32)
    img[:, 100:] = 0.6                      # contour franc
    out = tone_rescue(img, ToneParams(shadows=0.8, auto=False))
    gain = luminance(out) / np.maximum(luminance(img), 1e-6)
    dark_side = gain[100, 60:95]
    # le gain doit etre constant du cote sombre : un halo le ferait chuter
    # a l'approche du contour
    assert float(dark_side.max() / dark_side.min()) < 1.05


def test_explicit_amount_overrides_auto(scene):
    out = tone_rescue(scene, ToneParams(shadows=0.5, auto=False))
    assert np.median(luminance(out)) > np.median(luminance(scene))
