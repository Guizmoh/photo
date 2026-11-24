"""Entrees hostiles : tailles degenerees, valeurs extremes, fichiers casses.

Ces cas ne se rencontrent pas tous les jours, mais quand ils arrivent le
logiciel doit s'arreter proprement ou passer son chemin -- jamais produire
des NaN, planter, ou polluer la console d'avertissements.
"""

import os
import warnings

import numpy as np
import pytest

import synth
from naturaw.deblur import DeblurParams, deblur_linear_rgb
from naturaw.denoise import DenoiseParams, denoise_linear_rgb
from naturaw.deskew import DeskewParams, deskew_image
from naturaw.pipeline import Settings, process_array
from naturaw.rawio import load_image, save_image
from naturaw.rescue import RescueParams, apply_rescue
from naturaw.sharpen import SharpenParams, sharpen_linear_rgb
from naturaw.tone import ToneParams, tone_rescue

HOSTILE = {
    "1x1": np.full((1, 1, 3), 0.5, np.float32),
    "8x8": np.full((8, 8, 3), 0.5, np.float32),
    "impair 31x17": np.random.default_rng(0).random((31, 17, 3)).astype(np.float32),
    "tout noir": np.zeros((96, 96, 3), np.float32),
    "tout blanc": np.ones((96, 96, 3), np.float32),
    "une colonne": np.random.default_rng(1).random((128, 1, 3)).astype(np.float32),
    "une ligne": np.random.default_rng(1).random((1, 128, 3)).astype(np.float32),
    "bande fine": np.random.default_rng(1).random((4, 200, 3)).astype(np.float32),
}
STEPS = {
    "debruitage": lambda a: denoise_linear_rgb(a, DenoiseParams()),
    "sauvetage": lambda a: apply_rescue(a, RescueParams(banding="hv"), 0.01, report={}),
    "nettete": lambda a: sharpen_linear_rgb(a, SharpenParams(amount=1.0), 0.01),
    "tonalite": lambda a: tone_rescue(a, ToneParams(shadows=0.8, auto=False)),
    "deflou": lambda a: deblur_linear_rgb(a, DeblurParams(mode="force"), 0.01),
    "redressement": lambda a: deskew_image(a, DeskewParams()),
    "chaine": lambda a: process_array(a.copy(), Settings(), report={}),
}


@pytest.mark.parametrize("name", list(HOSTILE))
@pytest.mark.parametrize("step", list(STEPS))
def test_hostile_inputs_stay_finite_and_quiet(name, step):
    arr = HOSTILE[name]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)   # aucun avertissement numpy
        out = np.asarray(STEPS[step](arr))
    assert out.ndim == 3 and out.shape[2] == 3
    assert np.all(np.isfinite(out))


def test_extreme_values_do_not_explode():
    for scale, offset in ((6.0, 0.0), (1.0, -0.3), (0.002, 0.0)):
        img = (synth.architecture(120, 180) * scale + offset).astype(np.float32)
        out = process_array(img.copy(), Settings(), report={})
        assert np.all(np.isfinite(out))
        assert float(out.min()) >= 0.0


def test_hot_pixel_on_a_flat_image():
    img = np.full((128, 128, 3), 0.2, np.float32)
    img[64, 64] = 5.0
    out = process_array(img.copy(), Settings(), report={})
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("fmt,ext", [("tiff16", ".tif"), ("jpeg", ".jpg"), ("png", ".png")])
def test_accented_paths_and_metadata(tmp_path, fmt, ext):
    """Un nom de fichier accentue ne doit pas faire echouer l'ecriture.

    Le TIFF n'accepte que de l'ASCII 7 bits dans sa description : il a fallu
    echapper les accents du rapport de traitement.
    """
    img = synth.architecture(80, 120)
    path = str(tmp_path / f"éphémère été (1){ext}")
    save_image(path, img, fmt=fmt,
               metadata={"source": "éphémère.dng", "note": "redressée de +2°"})
    assert os.path.getsize(path) > 0
    assert load_image(path).rgb.shape[2] == 3


def test_broken_files_raise_clearly(tmp_path):
    empty = tmp_path / "vide.dng"
    empty.write_bytes(b"")
    with pytest.raises(Exception):
        load_image(str(empty))
    junk = tmp_path / "faux.jpg"
    junk.write_text("ceci n'est pas une image", encoding="utf-8")
    with pytest.raises(Exception):
        load_image(str(junk))
    with pytest.raises(FileNotFoundError):
        load_image(str(tmp_path / "absent.cr2"))


def test_unusual_image_formats_load(tmp_path):
    import cv2
    scene = synth.architecture(120, 180)
    grey = (np.clip(scene.mean(-1), 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(tmp_path / "gris.png"), grey)
    assert load_image(str(tmp_path / "gris.png")).rgb.shape[2] == 3
    rgba = np.dstack([(np.clip(scene, 0, 1) * 255).astype(np.uint8),
                      np.full(scene.shape[:2], 255, np.uint8)])
    cv2.imwrite(str(tmp_path / "alpha.png"), rgba)
    assert load_image(str(tmp_path / "alpha.png")).rgb.shape[2] == 3
