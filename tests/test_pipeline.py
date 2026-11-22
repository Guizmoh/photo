import json
import os

import numpy as np
import pytest

import synth
from naturaw.cli import build_parser, main, settings_from_args
from naturaw.deblur import convolve, linear_motion_kernel
from naturaw.deskew import rotate_image
from naturaw.pipeline import (Settings, collect_inputs, process_array,
                              process_file, summarize)
from naturaw.rawio import load_image


@pytest.fixture(scope="module")
def degraded(tmp_path_factory):
    """RAW de synthese : penche de -2.3 deg, bruite, legerement flou."""
    folder = tmp_path_factory.mktemp("photos")
    clean = synth.architecture(640, 960)
    tilted = rotate_image(clean, -2.3, crop="none")
    blurred = np.stack([convolve(tilted[..., c], linear_motion_kernel(9, 25))
                        for c in range(3)], -1)
    noisy = synth.add_photon_noise(blurred, 1.5e-3, 8e-5, seed=11)
    path = str(folder / "penchee.dng")
    synth.write_dng(path, noisy)
    return folder, path, clean


def test_end_to_end_straightens_and_denoises(degraded, tmp_path):
    folder, path, _clean = degraded
    out = str(tmp_path / "resultat.tif")
    report = process_file(path, out, Settings(color_space="raw"))
    assert os.path.exists(out)
    assert report["raw"] is True
    assert "+2.3" in report["deskew"] or "+2.2" in report["deskew"]
    assert report["noise_luma_sigma"] > 0.002
    assert "x" in report["dimensions_sortie"]
    assert isinstance(summarize(report), str)


def test_output_is_cleaner_than_the_raw_development(degraded, tmp_path):
    folder, path, _clean = degraded
    out = str(tmp_path / "net.tif")
    process_file(path, out, Settings(color_space="raw"))
    import cv2
    raw = load_image(path, color_space="raw").rgb
    done = load_image(out).rgb

    def grain(img):
        """Bruit haute frequence d'une zone de ciel, insensible au recadrage."""
        sky = img[20:120, 20:300]
        return float(np.std(sky - cv2.GaussianBlur(sky, (0, 0), 2.0)))

    assert grain(done) < grain(raw) * 0.6


def _all_off() -> Settings:
    s = Settings()
    s.denoise.strength = 0.0
    s.deblur.mode = "off"
    s.deskew.enabled = False
    s.sharpen.amount = 0.0
    s.tone.auto = False
    s.rescue.impulse = "off"
    s.exposure = "non"
    return s


def test_process_array_respects_disabled_steps():
    img = np.clip(synth.architecture(240, 360), 0.01, 1)
    report = {}
    out = process_array(img.copy(), _all_off(), report=report)
    assert np.allclose(out, img, atol=1e-6)
    assert report["debruitage"] == "desactive"


def test_default_settings_do_act_on_a_degraded_image():
    """Les reglages par defaut doivent reellement agir, pas se defiler."""
    clean = np.clip(synth.architecture(300, 450), 0.01, 1)
    noisy = synth.add_photon_noise(clean, 2.0e-3, 1.0e-4, seed=3)
    report = {}
    out = process_array(noisy.copy(), Settings(), report=report)
    assert not np.allclose(out, noisy, atol=1e-4)
    assert report["noise_luma_sigma"] > 0.001
    assert "nettete" in report


def test_exposure_modes():
    img = np.full((64, 64, 3), 0.1, np.float32)
    s = _all_off()
    s.exposure = "1.0"
    out = process_array(img.copy(), s, report={})
    assert abs(float(out.mean()) - 0.2) < 1e-4


def test_collect_inputs(degraded):
    folder, path, _ = degraded
    found = collect_inputs([str(folder)])
    assert path in found
    assert collect_inputs([path]) == [path]
    assert collect_inputs([str(folder)], raw_only=True) == [path]


def test_cli_list_mode(degraded, capsys):
    folder, path, _ = degraded
    code = main([str(folder), "--liste"])
    assert code == 0
    assert "penchee.dng" in capsys.readouterr().out


def test_cli_processes_and_writes_report(degraded, tmp_path):
    folder, path, _ = degraded
    rapport = str(tmp_path / "rapport.json")
    code = main([path, "-o", str(tmp_path), "-f", "jpeg", "-q", "90",
                 "--espace", "raw", "--rapport", rapport, "--cote-max", "700"])
    assert code == 0
    produced = [f for f in os.listdir(tmp_path) if f.endswith(".jpg")]
    assert produced == ["penchee_naturaw.jpg"]
    data = json.loads(open(rapport, encoding="utf-8").read())
    assert data and data[0]["sortie"].endswith(".jpg")


def test_cli_rejects_unknown_denoise_level():
    parser = build_parser()
    args = parser.parse_args(["x.dng", "-d", "nimportequoi"])
    with pytest.raises(SystemExit):
        settings_from_args(args)


def test_cli_preset_mapping():
    parser = build_parser()
    assert settings_from_args(parser.parse_args(["x", "-d", "fort"])).denoise.strength == 1.5
    assert settings_from_args(parser.parse_args(["x", "-d", "0.8"])).denoise.strength == 0.8
    s = settings_from_args(parser.parse_args(["x", "--apercu"]))
    assert s.half_size and s.max_side == 1800 and s.fmt == "jpeg"


def test_webui_settings_mapping():
    from naturaw.webui import _settings_from_payload
    s = _settings_from_payload({"debruitage": "leger", "grain": 0.3, "deflou": "on",
                                "deflou_longueur": 12, "recadrage": "max",
                                "format": "jpeg", "qualite": 80})
    assert s.denoise.strength == 0.6 and s.denoise.grain == 0.3
    assert s.deblur.mode == "on" and s.deblur.length == 12
    assert s.deskew.crop == "max" and s.fmt == "jpeg" and s.quality == 80
