import numpy as np

from naturaw.colorspace import rgb_to_ycocg
from naturaw.noise import estimate_scale_noise, measure_profile


def _noisy(a, b, seed=0, h=420, w=560):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.05, 0.95, w, dtype=np.float32)[None, :].repeat(h, 0)
    clean = np.dstack([x * 0.9, x, x * 0.75]).astype(np.float32)
    sigma = np.sqrt(a * clean + b).astype(np.float32)
    return clean, np.clip(clean + rng.standard_normal(clean.shape).astype(np.float32) * sigma,
                          0, None).astype(np.float32)


def test_model_recovers_signal_dependent_noise():
    a, b = 4.0e-4, 1.0e-5
    _clean, noisy = _noisy(a, b)
    prof = measure_profile(rgb_to_ycocg(noisy), levels=4)
    model = prof.channels[0][0]          # luminance, echelle la plus fine
    # variance attendue sur Y, attenuee par le filtre starlet de 1ere echelle
    expected_a = a * (0.9 / 16 + 1 / 4 + 0.75 / 16) * 0.8907 ** 2
    assert 0.5 * expected_a < model.a < 2.0 * expected_a
    assert model.sigma_at(0.5) > model.sigma_at(0.1)   # bruit croissant avec le signal


def test_noise_free_image_measures_almost_zero():
    x = np.linspace(0, 1, 400, dtype=np.float32)[None, :].repeat(300, 0)
    clean = np.dstack([x, x, x]).astype(np.float32)
    prof = measure_profile(rgb_to_ycocg(clean), levels=4)
    assert prof.luma_sigma(0.18) < 1e-3


def test_noise_decreases_with_scale():
    _clean, noisy = _noisy(1e-3, 5e-5, seed=2)
    models = estimate_scale_noise(rgb_to_ycocg(noisy)[..., 0],
                                  np.clip(rgb_to_ycocg(noisy)[..., 0], 0, None), levels=4)
    sig = [m.sigma_at(0.5) for m in models]
    assert all(sig[i + 1] <= sig[i] + 1e-9 for i in range(len(sig) - 1))


def test_profile_extends_to_requested_levels():
    _clean, noisy = _noisy(1e-3, 5e-5, seed=4)
    prof = measure_profile(rgb_to_ycocg(noisy), levels=6, measure_levels=3)
    assert all(len(ch) == 6 for ch in prof.channels)
