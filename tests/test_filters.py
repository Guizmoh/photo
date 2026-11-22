import numpy as np

from naturaw.filters import (box, cross_bilateral, guided_filter, local_range,
                             starlet_decompose, starlet_smooth)


def _img(seed=1):
    rng = np.random.default_rng(seed)
    base = np.add.outer(np.linspace(0, 1, 96), np.linspace(0, 0.5, 128)).astype(np.float32)
    return (base + 0.05 * rng.standard_normal((96, 128))).astype(np.float32)


def test_starlet_reconstruction_is_exact():
    img = _img()
    for levels in (1, 3, 5):
        details, residual = starlet_decompose(img, levels)
        rebuilt = residual + sum(details)
        assert np.allclose(img, rebuilt, atol=1e-5)


def test_starlet_smoothing_reduces_variance():
    img = _img()
    assert np.var(starlet_smooth(img, 0)) < np.var(img)


def test_box_filter_preserves_mean():
    img = _img()
    assert abs(float(box(img, 3).mean()) - float(img.mean())) < 1e-3


def test_cross_bilateral_output_stays_inside_local_range():
    """Point cle : une combinaison convexe ne peut pas inventer de valeur."""
    rng = np.random.default_rng(3)
    guide = np.zeros((64, 64), np.float32)
    guide[:, 32:] = 1.0
    src = np.zeros((64, 64), np.float32)
    src[:, 32:] = 0.8
    src += 0.02 * rng.standard_normal(src.shape).astype(np.float32)
    out = cross_bilateral(guide, src, radius=4, sigma_range=0.1)
    lo, hi = local_range(src, 4)
    assert np.all(out >= lo - 1e-5) and np.all(out <= hi + 1e-5)


def test_cross_bilateral_follows_guide_edges():
    guide = np.zeros((48, 48), np.float32)
    guide[:, 24:] = 1.0
    src = guide * 0.5
    out = cross_bilateral(guide, src, radius=5, sigma_range=0.15)
    # le contour reste franc : pas de fuite d'un cote a l'autre
    assert out[24, 20] < 0.06 and out[24, 28] > 0.44


def test_guided_filter_smooths_noise():
    rng = np.random.default_rng(5)
    guide = np.linspace(0, 1, 64, dtype=np.float32)[None, :].repeat(64, 0)
    noisy = guide + 0.1 * rng.standard_normal((64, 64)).astype(np.float32)
    out = guided_filter(guide, noisy, 4, 1e-4)
    assert np.std(out - guide) < np.std(noisy - guide) / 2
