"""Tests sur des captures simulees : scene -> mosaique Bayer -> bruit -> DNG.

C'est le test le plus proche d'un vrai fichier que l'on puisse faire sans
appareil : le bruit est ajoute au niveau du photosite, avant dematricage,
donc il est correle spatialement et colore comme sur une vraie photo.
"""

import os

import numpy as np
import pytest

import synth
from naturaw.colorspace import linear_to_srgb, rgb_to_ycocg
from naturaw.denoise import DenoiseParams, denoise_linear_rgb
from naturaw.noise import measure_profile
from naturaw.pyramid import analyse, pyramid_levels
from naturaw.rawio import load_image

# sensibilites approximatives, en electrons a saturation
ISO = {"200": 20000.0, "1600": 2500.0, "6400": 600.0, "25600": 150.0}


@pytest.fixture(scope="module")
def captures(tmp_path_factory):
    folder = tmp_path_factory.mktemp("captures")
    # taille realiste : la mesure du bruit par blocs est calibree pour des
    # fichiers d'appareil, et se laisse davantage tromper par la texture
    # quand l'image est petite (voir README, « Limites »)
    scenes = {"dead_leaves": synth.dead_leaves(900, 1300, seed=1),
              "architecture": synth.architecture(900, 1300)}
    out = {}
    for name, scene in scenes.items():
        ref_path = str(folder / f"{name}_ref.dng")
        synth.write_dng_mosaic(ref_path, synth.simulate_raw_capture(
            scene, full_well_e=1e9, read_noise_e=0.0, seed=1))
        shots = {}
        for iso, well in ISO.items():
            p = str(folder / f"{name}_{iso}.dng")
            synth.write_dng_mosaic(p, synth.simulate_raw_capture(
                scene, full_well_e=well, read_noise_e=3.0, seed=3))
            shots[iso] = load_image(p, color_space="raw").rgb
        out[name] = (load_image(ref_path, color_space="raw").rgb, shots)
    return out


def test_decoding_does_not_depend_on_the_brightest_pixel(captures, tmp_path):
    """LibRaw recale le blanc sur le pixel le plus clair : on doit l'avoir coupe.

    Sans cela, deux photos de la meme scene sortent avec des expositions
    differentes selon le bruit ou un reflet.
    """
    scene = synth.architecture(300, 450)
    quiet = str(tmp_path / "quiet.dng")
    loud = str(tmp_path / "loud.dng")
    synth.write_dng_mosaic(quiet, synth.simulate_raw_capture(
        scene, full_well_e=20000.0, read_noise_e=3.0, seed=2))
    mosaic = synth.simulate_raw_capture(scene, full_well_e=20000.0, read_noise_e=3.0, seed=2)
    mosaic[0, 0] = 1.0                      # un unique pixel au maximum
    synth.write_dng_mosaic(loud, mosaic)
    a = load_image(quiet, color_space="raw").rgb
    b = load_image(loud, color_space="raw").rgb
    assert abs(float(a.mean()) - float(b.mean())) / float(a.mean()) < 0.01


@pytest.mark.parametrize("iso", ["6400", "25600"])
def test_high_iso_captures_are_improved(captures, iso):
    for name, (ref, shots) in captures.items():
        rs = linear_to_srgb(ref)
        noisy = shots[iso]
        out = denoise_linear_rgb(noisy, DenoiseParams())
        ns, os_ = linear_to_srgb(noisy), linear_to_srgb(out)
        assert synth.psnr(rs, os_) > synth.psnr(rs, ns), name
        assert synth.ssim(rs.mean(-1), os_.mean(-1)) > \
               synth.ssim(rs.mean(-1), ns.mean(-1)) + 0.03, name


def test_base_iso_photo_is_barely_touched(captures):
    """A 200 ISO le bruit ne se voit pas : il ne faut rien abimer."""
    ref, shots = captures["architecture"]
    out = denoise_linear_rgb(shots["200"], DenoiseParams())
    rs = linear_to_srgb(ref)
    before = synth.psnr(rs, linear_to_srgb(shots["200"]))
    after = synth.psnr(rs, linear_to_srgb(out))
    assert after >= before - 0.1


def test_texture_is_preserved_on_a_dead_leaves_target(captures):
    """MTF de texture : la mesure de l'industrie photo, par spectre croise."""
    ref, shots = captures["dead_leaves"]
    rs = linear_to_srgb(ref)
    out = linear_to_srgb(denoise_linear_rgb(shots["6400"], DenoiseParams()))
    mtf = synth.texture_mtf(out, rs)
    assert min(mtf.values()) > 0.85, mtf
    strong = linear_to_srgb(denoise_linear_rgb(shots["6400"],
                                               DenoiseParams(strength=2.1)))
    # un debruitage plus fort doit se payer en texture : la mesure doit le voir
    assert min(synth.texture_mtf(strong, rs).values()) < min(mtf.values())


def test_noise_model_matches_the_real_noise(captures):
    """Le modele mesure doit coller au bruit reel, meme sur une texture dense."""
    for name, (ref, shots) in captures.items():
        noisy = shots["6400"]
        levels = pyramid_levels(noisy.shape)
        d_noisy, _ = analyse(rgb_to_ycocg(noisy)[..., 0], levels)
        d_ref, _ = analyse(rgb_to_ycocg(ref)[..., 0], levels)
        prof = measure_profile(rgb_to_ycocg(noisy), levels)
        for j in range(3):
            true_sigma = float(np.std(d_noisy[j] - d_ref[j]))
            est = float(np.sqrt(np.mean(prof.channels[0][j].variance(
                np.clip(rgb_to_ycocg(noisy)[..., 0], 0, None)))))
            ratio = est / max(true_sigma, 1e-9)
            # jamais trop haut (on effacerait du detail), jamais trop bas
            # (on laisserait du bruit)
            assert 0.35 < ratio < 2.2, (name, j, ratio)


def test_dead_leaves_has_natural_image_statistics():
    """La mire doit bien avoir un spectre en 1/f^2, sinon elle ne prouve rien."""
    img = synth.dead_leaves(400, 600, seed=3).mean(-1)
    spec = np.abs(np.fft.fftshift(np.fft.fft2(img - img.mean()))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(img.shape[0]))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(img.shape[1]))[None, :]
    rho = np.sqrt(fy ** 2 + fx ** 2)
    sel = (rho > 0.015) & (rho < 0.4)
    slope = np.polyfit(np.log(rho[sel]), np.log(spec[sel]), 1)[0]
    assert -2.6 < slope < -1.5


def test_simulated_noise_is_spatially_correlated():
    """Le bruit d'un vrai fichier est correle par le dematricage ; le notre aussi."""
    scene = synth.dead_leaves(400, 600, seed=5)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.dng")
        b = os.path.join(d, "b.dng")
        synth.write_dng_mosaic(a, synth.simulate_raw_capture(scene, 600.0, 3.0, seed=7))
        synth.write_dng_mosaic(b, synth.simulate_raw_capture(scene, 1e9, 0.0, seed=7))
        err = (load_image(a, color_space="raw").rgb -
               load_image(b, color_space="raw").rgb)[..., 1]
    err = err - err.mean()
    corr = float((err[:, :-1] * err[:, 1:]).mean() / err.var())
    assert corr > 0.05      # un bruit blanc donnerait 0,00
