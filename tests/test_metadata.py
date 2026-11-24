"""Les metadonnees de prise de vue doivent survivre au traitement.

Une photo developpee qui a perdu sa date et son boitier devient difficile a
classer : les catalogueurs trient sur ces champs.
"""

import os

import cv2
import numpy as np
import pytest

import synth
from naturaw.metadata import (CaptureInfo, attach_to_jpeg, exif_payload,
                              read_capture_info, tiff_extratags)
from naturaw.pipeline import Settings, process_file

piexif = pytest.importorskip("piexif")


@pytest.fixture()
def source(tmp_path):
    """Un JPEG muni d'un EXIF complet, comme sortirait d'un appareil."""
    path = str(tmp_path / "prise.jpg")
    img = (np.clip(synth.architecture(300, 450), 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    exif = {
        "0th": {piexif.ImageIFD.Make: "Nikon", piexif.ImageIFD.Model: "Z 8",
                piexif.ImageIFD.Artist: "Guizmoh",
                piexif.ImageIFD.Copyright: "(c) 2025"},
        "Exif": {piexif.ExifIFD.DateTimeOriginal: "2025:06:21 11:04:33",
                 piexif.ExifIFD.ISOSpeedRatings: 3200,
                 piexif.ExifIFD.ExposureTime: (1, 400),
                 piexif.ExifIFD.FNumber: (18, 10),
                 piexif.ExifIFD.FocalLength: (35, 1),
                 piexif.ExifIFD.LensModel: "NIKKOR Z 35mm f/1.8"},
        "GPS": {piexif.GPSIFD.GPSLatitudeRef: "N",
                piexif.GPSIFD.GPSLatitude: ((48, 1), (51, 1), (2976, 100)),
                piexif.GPSIFD.GPSLongitudeRef: "E",
                piexif.GPSIFD.GPSLongitude: ((2, 1), (17, 1), (2808, 100))},
        "1st": {}, "thumbnail": None,
    }
    piexif.insert(piexif.dump(exif), path)
    return path


def test_capture_info_is_read(source):
    info = read_capture_info(source)
    assert info
    assert info.fields["boitier"] == "Z 8"
    assert str(info.fields["date"]).startswith("2025:06:21")
    assert info.gps, "la position GPS doit etre lue"
    resume = info.summary()
    assert "Z 8" in resume and "3200 ISO" in resume and "f/1.8" in resume


def test_jpeg_output_keeps_the_exif(source, tmp_path):
    out = str(tmp_path / "sortie.jpg")
    report = process_file(source, out, Settings(fmt="jpeg"))
    assert "Z 8" in report["prise_de_vue"]
    back = piexif.load(out)
    assert back["0th"][piexif.ImageIFD.Model] == b"Z 8"
    assert back["0th"][piexif.ImageIFD.Artist] == b"Guizmoh"
    assert back["Exif"][piexif.ExifIFD.DateTimeOriginal] == b"2025:06:21 11:04:33"
    assert back["Exif"][piexif.ExifIFD.ISOSpeedRatings] == 3200
    assert back["Exif"][piexif.ExifIFD.LensModel] == b"NIKKOR Z 35mm f/1.8"
    assert back["GPS"], "la position GPS doit etre recopiee"
    assert b"naturaw" in back["0th"][piexif.ImageIFD.Software]
    # l'orientation est remise a 1 : l'image est deja redressee
    assert back["0th"].get(piexif.ImageIFD.Orientation, 1) == 1


def test_tiff_output_keeps_the_main_tags(source, tmp_path):
    import tifffile
    out = str(tmp_path / "sortie.tif")
    process_file(source, out, Settings(fmt="tiff16"))
    with tifffile.TiffFile(out) as tif:
        tags = tif.pages[0].tags
        assert tags["Make"].value == "Nikon"
        assert tags["Model"].value == "Z 8"
        assert tags["DateTime"].value.startswith("2025:06:21")
        assert tags["Artist"].value == "Guizmoh"


def test_file_without_exif_is_processed_anyway(tmp_path):
    path = str(tmp_path / "nu.png")
    cv2.imwrite(path, np.full((80, 120, 3), 128, np.uint8))
    out = str(tmp_path / "sortie.jpg")
    report = process_file(path, out, Settings(fmt="jpeg"))
    assert os.path.exists(out)
    assert not report.get("prise_de_vue")


def test_missing_values_do_not_break_the_exif_block():
    info = CaptureInfo(fields={"boitier": "X", "vitesse": "pas un nombre",
                               "iso": None, "ouverture": ""})
    payload = exif_payload(info)
    assert payload is None or isinstance(payload, bytes)
    assert attach_to_jpeg("/chemin/inexistant.jpg", payload) is False
    assert attach_to_jpeg("/chemin/inexistant.jpg", None) is False
    assert isinstance(tiff_extratags(info), list)
