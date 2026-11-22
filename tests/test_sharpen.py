import cv2
import numpy as np

import synth
from naturaw.colorspace import linear_to_srgb
from naturaw.deblur import convolve
from naturaw.sharpen import SharpenParams, inverse_kernel, sharpen_linear_rgb


def _soft(scene, sigma=0.75):
    k = cv2.getGaussianKernel(9, sigma).astype(np.float32)
    kern = (k @ k.T).astype(np.float32)
    return np.stack([convolve(scene[..., c], kern) for c in range(3)], -1)


def test_inverse_kernel_is_a_sharpening_filter():
    k = inverse_kernel(0.8, 0.02)
    assert abs(float(k.sum()) - 1.0) < 1e-5          # moyenne preservee
    assert k[k.shape[0] // 2, k.shape[1] // 2] > 1.0  # centre amplifie
    assert k.min() < 0.0                              # lobes negatifs


def test_sharpening_recovers_capture_softness():
    clean = synth.architecture(400, 600)
    soft = _soft(clean)
    out = sharpen_linear_rgb(soft, SharpenParams(amount=0.8), noise_sigma=0.002)
    c, s, o = (linear_to_srgb(x) for x in (clean, soft, out))
    assert synth.psnr(c, o) > synth.psnr(c, s) + 1.0
    assert synth.gradient_energy(o.mean(-1)) > synth.gradient_energy(s.mean(-1))


def test_beats_a_plain_unsharp_mask():
    clean = synth.architecture(400, 600)
    soft = synth.add_photon_noise(_soft(clean), 1.2e-3, 6e-5, seed=5)
    mine = sharpen_linear_rgb(soft, SharpenParams(), noise_sigma=0.012)
    naive = np.clip(soft + 0.8 * (soft - cv2.GaussianBlur(soft, (0, 0), 1.0)), 0, None)
    c = linear_to_srgb(clean)
    assert synth.psnr(c, linear_to_srgb(mine)) > synth.psnr(c, linear_to_srgb(naive))


def test_flat_noisy_areas_are_not_amplified():
    flat = np.full((200, 200, 3), 0.25, np.float32)
    noisy = synth.add_photon_noise(flat, 1.5e-3, 1e-4, seed=8)
    out = sharpen_linear_rgb(noisy, SharpenParams(amount=1.0), noise_sigma=0.02)
    assert float(np.std(out)) < float(np.std(noisy)) * 1.08


def test_amount_zero_is_identity():
    img = synth.architecture(120, 160)
    assert sharpen_linear_rgb(img, SharpenParams(amount=0.0)) is img


def test_no_overshoot_beyond_local_range():
    img = np.full((120, 160, 3), 0.2, np.float32)
    img[:, 80:] = 0.7
    out = sharpen_linear_rgb(img, SharpenParams(amount=1.2), noise_sigma=0.001)
    assert float(out.max()) < 0.78 and float(out.min()) > 0.14
