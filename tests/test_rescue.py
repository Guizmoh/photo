import numpy as np

import synth
from naturaw.colorspace import linear_to_srgb
from naturaw.rescue import (RescueParams, apply_rescue, measure_banding,
                            remove_banding, remove_impulse)


def _scene():
    clean = synth.architecture(400, 600)
    return clean, synth.add_photon_noise(clean, 1.5e-3, 8e-5, seed=5)


def test_hot_pixels_are_removed():
    _clean, noisy = _scene()
    rng = np.random.default_rng(4)
    hot = noisy.copy()
    n = 300
    ys, xs = rng.integers(0, 400, n), rng.integers(0, 600, n)
    hot[ys, xs] = np.clip(hot[ys, xs] + rng.uniform(0.3, 0.9, (n, 1)), 0, 2)
    report = {}
    fixed = remove_impulse(hot, 0.012, report=report)
    assert "pixels" in report["pixels_chauds"]
    before = synth.psnr(linear_to_srgb(noisy), linear_to_srgb(hot))
    after = synth.psnr(linear_to_srgb(noisy), linear_to_srgb(fixed))
    assert after > before + 10.0


def test_clean_image_is_not_damaged_by_impulse_filter():
    _clean, noisy = _scene()
    out = remove_impulse(noisy, 0.012)
    assert synth.psnr(linear_to_srgb(noisy), linear_to_srgb(out)) > 55.0


def test_fine_detail_survives_impulse_filter():
    """Un detail fin contraste n'est pas un pixel chaud."""
    img = np.full((128, 128, 3), 0.2, np.float32)
    img[40:44, 40:44] = 0.9           # petit carre de 4 px
    img[60, 20:100] = 0.85            # ligne fine
    out = remove_impulse(img, 0.004)
    assert abs(float(out[41, 41, 0]) - 0.9) < 0.02
    assert abs(float(out[60, 60, 0]) - 0.85) < 0.02


def test_banding_is_detected_and_reduced():
    tex = np.clip(np.dstack([synth.texture(400, 600, s) * 0.5 + 0.05
                             for s in (1, 2, 3)]), 0.01, 1).astype(np.float32)
    base = synth.add_photon_noise(tex, 1.5e-3, 8e-5, seed=6)
    rng = np.random.default_rng(2)
    banded = np.clip(base + rng.normal(0, 0.012, (400, 1, 1)).astype(np.float32), 0, None)
    rows, cols = measure_banding(banded.mean(axis=2))
    assert rows > 0.3 and rows > cols
    fixed = remove_banding(banded, mode="h")
    assert synth.psnr(linear_to_srgb(base), linear_to_srgb(fixed)) > \
           synth.psnr(linear_to_srgb(base), linear_to_srgb(banded)) + 1.0


def test_banding_off_by_default():
    _clean, noisy = _scene()
    assert remove_banding(noisy) is noisy
    out = apply_rescue(noisy, RescueParams(), 0.012, report={})
    assert synth.psnr(linear_to_srgb(noisy), linear_to_srgb(out)) > 55.0
