import numpy as np

from naturaw.colorspace import (apply_gamma, linear_to_srgb, luminance,
                                rgb_to_ycocg, srgb_to_linear, ycocg_to_rgb)


def test_ycocg_roundtrip_exact():
    rng = np.random.default_rng(0)
    rgb = rng.random((64, 48, 3), dtype=np.float32)
    back = ycocg_to_rgb(rgb_to_ycocg(rgb))
    assert np.allclose(rgb, back, atol=1e-6)


def test_ycocg_luma_is_grey_for_grey():
    grey = np.full((8, 8, 3), 0.42, np.float32)
    ycc = rgb_to_ycocg(grey)
    assert np.allclose(ycc[..., 0], 0.42, atol=1e-6)
    assert np.allclose(ycc[..., 1:], 0.0, atol=1e-6)


def test_srgb_roundtrip():
    x = np.linspace(0, 1, 2048, dtype=np.float32)
    assert np.allclose(x, srgb_to_linear(linear_to_srgb(x)), atol=2e-5)


def test_srgb_is_monotonic_and_bounded():
    x = np.linspace(0, 1, 512, dtype=np.float32)
    y = linear_to_srgb(x)
    assert np.all(np.diff(y) >= -1e-7)
    assert y.min() >= 0.0 and y.max() <= 1.0


def test_apply_gamma_modes():
    x = np.full((4, 4), 0.18, np.float32)
    assert apply_gamma(x, "linear")[0, 0] == np.float32(0.18)
    assert apply_gamma(x, "srgb")[0, 0] > 0.4
    assert apply_gamma(x, "gamma22")[0, 0] > 0.4


def test_luminance_weights():
    red = np.zeros((2, 2, 3), np.float32)
    red[..., 0] = 1.0
    assert np.allclose(luminance(red), 0.2126, atol=1e-6)
