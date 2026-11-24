import os

import numpy as np
import pytest

import synth
from naturaw.rawio import (auto_exposure, default_output_name, is_raw_file,
                           is_supported, load_image, save_image)


@pytest.fixture(scope="module")
def dng(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("raw") / "essai.dng")
    scene = synth.architecture(300, 420)
    synth.write_dng(path, scene)
    return path, scene


def test_extension_detection():
    assert is_raw_file("/a/b/IMG_0001.CR2") and is_raw_file("x.nef")
    assert not is_raw_file("x.jpg")
    assert is_supported("x.jpg") and is_supported("x.ARW")
    assert not is_supported("x.txt")


def test_dng_is_decoded_in_linear_light(dng):
    path, scene = dng
    img = load_image(path, color_space="raw")
    assert img.is_raw and img.rgb.dtype == np.float32
    assert img.rgb.shape == scene.shape
    inner = (slice(16, -16), slice(16, -16))
    # le dematricage reste fidele a la scene d'origine
    assert synth.psnr(scene[inner], img.rgb[inner]) > 30.0


def test_half_size_decoding(dng):
    path, scene = dng
    img = load_image(path, half_size=True)
    assert img.rgb.shape[0] <= scene.shape[0] // 2 + 2


@pytest.mark.parametrize("fmt,ext", [("tiff16", ".tif"), ("png16", ".png"),
                                     ("jpeg", ".jpg"), ("png", ".png"),
                                     ("tiff8", ".tif")])
def test_write_and_reread(tmp_path, fmt, ext):
    rng = np.random.default_rng(2)
    img = np.clip(rng.random((64, 80, 3), dtype=np.float32), 0.01, 0.99)
    out = str(tmp_path / ("x" + ext))
    save_image(out, img, fmt=fmt, metadata={"test": True}, dither=False)
    assert os.path.getsize(out) > 0
    back = load_image(out)
    assert back.rgb.shape == img.shape
    tol = 0.02 if fmt in ("jpeg", "png", "tiff8") else 1e-3
    assert float(np.abs(back.rgb - img).mean()) < tol


def test_auto_exposure_is_bounded():
    dark = np.full((32, 32, 3), 0.01, np.float32)
    bright = np.full((32, 32, 3), 1.0, np.float32)
    assert 1.0 < auto_exposure(dark) <= 2.0 ** 3.5 + 1e-6
    assert auto_exposure(bright) >= 0.5


def test_auto_exposure_respects_the_mood_of_the_scene():
    """Deux contraintes : ne pas bruler, et ne pas reveiller une scene sombre."""
    rng = np.random.default_rng(0)
    base = rng.random((200, 300, 3)).astype(np.float32)

    snow = np.clip(base * 0.3 + 0.55, 0, 1)        # scene claire
    night = np.clip(base * 0.02, 0, 1)             # scene nocturne sans lumiere vive
    night[20:30, 40:60] = 0.9                      # un lampadaire
    under = np.clip(base * 0.5 * 0.08, 0, 1)       # vraie sous-exposition

    # la neige ne doit pas etre grisee, ni brulee
    g_snow = auto_exposure(snow)
    assert 0.9 <= g_snow <= 1.2
    # la nuit reste la nuit : le lampadaire interdit de tout eclaircir
    assert auto_exposure(night) < 1.3
    # la photo sous-exposee est remontee franchement
    g_under = auto_exposure(under)
    assert g_under > 3.0
    assert float(np.percentile(under * g_under, 99.7)) <= 1.0


def test_output_naming():
    assert default_output_name("/x/y/IMG.CR2", "_naturaw", "tiff16") == "IMG_naturaw.tif"
    assert default_output_name("IMG.NEF", "_v2", "jpeg") == "IMG_v2.jpg"
