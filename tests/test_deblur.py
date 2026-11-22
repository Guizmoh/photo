import numpy as np
import pytest

import synth
from naturaw.colorspace import linear_to_srgb, luminance
from naturaw.deblur import (DeblurParams, analyze_blur, convolve, correlate,
                            deblur_linear_rgb, delta_kernel, kernel_moments,
                            linear_motion_kernel, normalize_kernel,
                            prepare_crops, richardson_lucy)


@pytest.fixture(scope="module")
def scene():
    return synth.architecture(560, 820)


def test_linear_kernel_geometry():
    for length, angle in ((9, 20), (15, 70), (21, 135)):
        k = linear_motion_kernel(length, angle)
        major, minor, aniso, est_angle = kernel_moments(k)
        assert abs(k.sum() - 1.0) < 1e-5
        assert abs(major - length / np.sqrt(12.0)) < 0.35   # ecart-type d'un segment
        assert minor < 0.6 and aniso > 3.0
        assert min(abs(est_angle - angle), 180 - abs(est_angle - angle)) < 1.5


def test_normalize_kernel_cleans_up():
    k = np.zeros((11, 11), np.float32)
    k[5, 4:7] = 1.0
    k[0, 0] = 0.4          # tache isolee parasite
    k[9, 9] = -2.0         # valeur negative impossible
    out = normalize_kernel(k, prune=0.05)
    assert out.min() >= 0.0
    assert abs(out.sum() - 1.0) < 1e-6
    assert out[0, 0] == 0.0 and out[9, 9] == 0.0


def test_convolve_and_correlate_are_adjoint():
    """<K a, b> = <a, K^T b> : indispensable pour que Richardson-Lucy converge.

    Les deux signaux sont nuls pres des bords, de sorte que la gestion des
    bords n'intervient pas dans le produit scalaire.
    """
    rng = np.random.default_rng(0)
    a = np.zeros((80, 88), np.float32)
    b = np.zeros((80, 88), np.float32)
    a[20:60, 24:64] = rng.standard_normal((40, 40))
    b[20:60, 24:64] = rng.standard_normal((40, 40))
    k = linear_motion_kernel(7, 33)
    left = float(np.sum(convolve(a, k) * b))
    right = float(np.sum(a * correlate(b, k)))
    assert abs(left - right) / max(abs(left), 1e-6) < 1e-4


def test_convolution_matches_fourier_reference():
    rng = np.random.default_rng(1)
    img = rng.standard_normal((64, 72)).astype(np.float32)
    k = linear_motion_kernel(9, 15)
    kh, kw = k.shape
    pad = np.zeros_like(img)
    pad[:kh, :kw] = k
    pad = np.roll(np.roll(pad, -(kh // 2), 0), -(kw // 2), 1)
    ref = np.fft.irfft2(np.fft.rfft2(img) * np.fft.rfft2(pad), s=img.shape)
    inner = (slice(16, -16), slice(16, -16))
    assert np.allclose(ref[inner], convolve(img, k)[inner], atol=1e-5)


def test_richardson_lucy_recovers_known_blur(scene):
    k = linear_motion_kernel(11, 30)
    blurred = convolve(luminance(scene), k)
    restored = richardson_lucy(blurred, k, iterations=25)
    ref = luminance(scene)
    inner = (slice(40, -40), slice(40, -40))
    assert synth.psnr(ref[inner], restored[inner]) > synth.psnr(ref[inner], blurred[inner]) + 3.0


def test_sharp_image_is_left_untouched(scene):
    noisy = synth.add_photon_noise(scene, 2e-4, 2e-5, seed=3)
    report = {}
    out = deblur_linear_rgb(noisy, DeblurParams(mode="auto"), report=report)
    assert out is noisy
    assert "ignore" in report["deblur"]


def test_defocus_is_not_treated_as_motion(scene):
    import cv2
    g = cv2.getGaussianKernel(13, 1.8).astype(np.float32)
    k = (g @ g.T).astype(np.float32)
    blurred = np.stack([convolve(scene[..., c], k) for c in range(3)], -1)
    est = analyze_blur(luminance(blurred), DeblurParams())
    assert not est.reliable


def test_manual_kernel_sharpens_a_known_blur(scene):
    k = linear_motion_kernel(13, 25)
    blurred = np.stack([convolve(scene[..., c], k) for c in range(3)], -1)
    blurred = synth.add_photon_noise(blurred, 2e-4, 2e-5, seed=4)
    params = DeblurParams(mode="force", length=13, angle=25, strength=0.9)
    out = deblur_linear_rgb(blurred, params, noise_sigma=0.004)
    c, b, o = (linear_to_srgb(x) for x in (scene, blurred, out))
    inner = (slice(40, -40), slice(40, -40))
    assert synth.ssim(c[inner].mean(-1), o[inner].mean(-1)) > \
           synth.ssim(c[inner].mean(-1), b[inner].mean(-1))
    assert synth.gradient_energy(o.mean(-1)) > synth.gradient_energy(b.mean(-1))


def test_blind_estimate_finds_direction_on_textured_scene():
    tex = np.clip(np.dstack([synth.texture(600, 820, s) * k + 0.05
                             for s, k in ((1, 0.6), (2, 0.55), (3, 0.45))]), 0.01, 1).astype(np.float32)
    k = linear_motion_kernel(13, 115)
    blurred = np.stack([convolve(tex[..., c], k) for c in range(3)], -1)
    est = analyze_blur(luminance(blurred), DeblurParams())
    assert est.extent > 2.0
    assert min(abs(est.angle - 115), 180 - abs(est.angle - 115)) < 25


def test_off_mode_returns_input(scene):
    assert deblur_linear_rgb(scene, DeblurParams(mode="off")) is scene


def test_prepare_crops_returns_windowed_patches(scene):
    crops = prepare_crops(luminance(scene), size=256, count=3)
    assert 1 <= len(crops) <= 3
    for patch, texture in crops:
        assert patch.shape == (256, 256)
        assert abs(float(patch[0, :].mean())) < 1e-6     # fenetre de Hann : bords nuls
        assert texture > 0.0


def test_delta_kernel_has_no_spread():
    assert kernel_moments(delta_kernel(5))[0] < 2e-3
