import numpy as np
import pytest

import synth
from naturaw.colorspace import linear_to_srgb, rgb_to_ycocg
from naturaw.denoise import DenoiseParams, denoise_linear_rgb
from naturaw.noise import measure_profile


@pytest.fixture(scope="module")
def scene():
    clean = synth.architecture(500, 760)
    noisy = synth.add_photon_noise(clean, a=1.8e-3, b=1.0e-4, seed=5)
    return clean, noisy


def test_denoise_improves_psnr_and_ssim(scene):
    clean, noisy = scene
    out = denoise_linear_rgb(noisy, DenoiseParams())
    c, n, o = (linear_to_srgb(x) for x in (clean, noisy, out))
    assert synth.psnr(c, o) > synth.psnr(c, n) + 3.0
    assert synth.ssim(c.mean(-1), o.mean(-1)) > synth.ssim(c.mean(-1), n.mean(-1)) + 0.05


def test_flat_areas_get_much_cleaner(scene):
    clean, noisy = scene
    out = denoise_linear_rgb(noisy, DenoiseParams())
    sky = (slice(10, 110), slice(10, 300))
    before = float(np.std(noisy[sky] - clean[sky]))
    after = float(np.std(out[sky] - clean[sky]))
    assert after < before * 0.45


def test_colour_noise_is_reduced_more_than_luma(scene):
    """La couleur est nettoyee plus fort que la luminance (detail moins utile)."""
    clean, noisy = scene
    out = denoise_linear_rgb(noisy, DenoiseParams())
    err_before = rgb_to_ycocg(noisy - clean)
    err_after = rgb_to_ycocg(out - clean)
    luma_gain = np.std(err_after[..., 0]) / np.std(err_before[..., 0])
    chroma_gain = max(np.std(err_after[..., c]) / np.std(err_before[..., c])
                      for c in (1, 2))
    assert chroma_gain < luma_gain


def test_texture_is_preserved(scene):
    """Le debruitage ne doit pas 'plastifier' les zones texturees.

    Reference de comparaison : un flou gaussien leger, qui enleve a peu pres
    autant de bruit dans les zones unies.
    """
    import cv2
    clean, noisy = scene
    out = denoise_linear_rgb(noisy, DenoiseParams())
    blurred = cv2.GaussianBlur(noisy, (0, 0), 1.0)
    reg = (slice(300, 480), slice(80, 700))

    def detail_match(img):
        """Correlation du detail fin avec la texture d'origine."""
        hp = lambda x: x - cv2.GaussianBlur(x, (0, 0), 1.2)
        a = hp(linear_to_srgb(clean)[reg].mean(-1))
        b = hp(linear_to_srgb(img)[reg].mean(-1))
        return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])

    # la texture reelle ressort nettement mieux qu'avant debruitage
    assert detail_match(out) > detail_match(noisy) + 0.05
    # et il reste de la matiere : pas de rendu lisse / plastique
    energy = synth.gradient_energy(linear_to_srgb(out)[reg].mean(-1))
    ref = synth.gradient_energy(linear_to_srgb(clean)[reg].mean(-1))
    assert energy > 0.4 * ref
    # sur l'image entiere, on bat largement un flou gaussien de force voisine
    c, o, g = (linear_to_srgb(x) for x in (clean, out, blurred))
    assert synth.psnr(c, o) > synth.psnr(c, g) + 3.0
    assert synth.ssim(c.mean(-1), o.mean(-1)) > synth.ssim(c.mean(-1), g.mean(-1))


def test_grain_setting_changes_residual_texture(scene):
    _clean, noisy = scene
    smooth = denoise_linear_rgb(noisy, DenoiseParams(grain=0.0))
    grainy = denoise_linear_rgb(noisy, DenoiseParams(grain=0.35))
    sky = (slice(10, 110), slice(10, 300))
    assert np.std(grainy[sky]) > np.std(smooth[sky])


def test_strength_zero_is_identity(scene):
    _clean, noisy = scene
    assert denoise_linear_rgb(noisy, DenoiseParams(strength=0.0)) is noisy


def test_safety_clamp_limits_deviation(scene):
    """Aucun pixel ne doit s'eloigner de l'original de plus de safety sigma."""
    _clean, noisy = scene
    params = DenoiseParams(safety=5.0)
    out = denoise_linear_rgb(noisy, params)
    from naturaw.filters import box
    ycc_in = rgb_to_ycocg(noisy)
    ycc_out = rgb_to_ycocg(out)
    prof = measure_profile(ycc_in, levels=5)
    bright = np.clip(box(ycc_in[..., 0], 3), 0, None)
    for c in range(3):
        limit = params.safety * np.sqrt(prof.channels[c][0].variance(bright))
        assert np.all(np.abs(ycc_out[..., c] - ycc_in[..., c]) <= limit + 1e-3)


def test_no_colour_shift_on_neutral_patch():
    grey = np.full((160, 160, 3), 0.25, np.float32)
    noisy = synth.add_photon_noise(grey, 1e-3, 5e-5, seed=9)
    out = denoise_linear_rgb(noisy, DenoiseParams())
    mean = out.reshape(-1, 3).mean(0)
    assert np.all(np.abs(mean - 0.25) < 0.01)
    assert float(np.abs(mean[0] - mean[2])) < 2e-3
