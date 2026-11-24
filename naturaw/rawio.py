"""Lecture des fichiers RAW et ecriture des resultats.

Le RAW est developpe en lumiere *lineaire* (gamma 1, pas d'eclaircissement
automatique, aucune reduction de bruit du decodeur) : c'est l'etat dans lequel
le modele de bruit du capteur et la deconvolution sont valables. La courbe de
transfert n'est appliquee qu'a l'ecriture.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import cv2
import numpy as np

from .colorspace import apply_gamma, srgb_to_linear
from .metadata import CaptureInfo, attach_to_jpeg, exif_payload, read_capture_info, tiff_extratags

RAW_EXTENSIONS = {
    ".3fr", ".arw", ".braw", ".cap", ".cr2", ".cr3", ".crw", ".dcr", ".dng",
    ".erf", ".fff", ".gpr", ".iiq", ".k25", ".kdc", ".mdc", ".mef", ".mos",
    ".mrw", ".nef", ".nrw", ".orf", ".ori", ".pef", ".raf", ".raw", ".rw2",
    ".rwl", ".sr2", ".srf", ".srw", ".x3f",
}
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".ppm"}

DEMOSAIC = {
    "ahd": "AHD", "vng": "VNG", "ppg": "PPG", "dcb": "DCB",
    "dht": "DHT", "aahd": "AAHD", "linear": "LINEAR",
}
COLOR_SPACES = {"srgb": "sRGB", "adobe": "Adobe", "prophoto": "ProPhoto",
                "wide": "Wide", "xyz": "XYZ", "raw": "raw"}


@dataclass
class LoadedImage:
    """Image en lumiere lineaire, float32, canaux RGB dans [0, ~1]."""

    rgb: np.ndarray
    path: str = ""
    is_raw: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)
    capture: CaptureInfo = field(default_factory=CaptureInfo)
    """Metadonnees de prise de vue, recopiees dans le fichier produit."""


def is_raw_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in RAW_EXTENSIONS


def is_supported(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in RAW_EXTENSIONS or ext in IMAGE_EXTENSIONS


def _read_exif(path: str) -> Dict[str, Any]:
    """Champs de prise de vue, sous forme lisible (pour le rapport)."""
    info = read_capture_info(path)
    return {k: str(v) for k, v in info.fields.items()}


def load_image(path: str, half_size: bool = False, demosaic: str = "ahd",
               camera_wb: bool = True, color_space: str = "srgb",
               highlight_mode: int = 0) -> LoadedImage:
    """Charge un RAW (ou une image deja developpee) en lumiere lineaire."""
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if is_raw_file(path):
        import rawpy
        meta: Dict[str, Any] = {}
        algo = DEMOSAIC.get(str(demosaic).lower(), "AHD")
        with rawpy.imread(path) as raw:
            meta["taille_capteur"] = tuple(int(v) for v in raw.raw_image.shape)
            try:
                meta["balance_boitier"] = [float(v) for v in raw.camera_whitebalance]
                meta["motif"] = str(raw.color_desc.decode() if isinstance(raw.color_desc, bytes)
                                    else raw.color_desc)
            except Exception:
                pass
            params = dict(
                output_bps=16,
                gamma=(1.0, 1.0),              # sortie lineaire
                no_auto_bright=True,           # pas d'eclaircissement cache
                use_camera_wb=bool(camera_wb),
                use_auto_wb=not bool(camera_wb),
                median_filter_passes=0,        # aucun lissage du decodeur
                output_color=getattr(rawpy.ColorSpace,
                                     COLOR_SPACES.get(str(color_space).lower(), "sRGB")),
                demosaic_algorithm=getattr(rawpy.DemosaicAlgorithm, algo),
                fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Off,
                highlight_mode=highlight_mode,
                half_size=bool(half_size),
                # Par defaut LibRaw abaisse le niveau de blanc jusqu'au pixel
                # le plus clair du fichier (seuil 0,75), ce qui fait varier
                # l'exposition d'une photo a l'autre selon le bruit ou un
                # reflet : mesure sur la meme scene, 0,279 contre 0,360 de
                # moyenne, soit 29 % d'ecart. On garde le niveau de blanc
                # declare, et c'est notre propre exposition automatique qui
                # decide de la luminosite.
                adjust_maximum_thr=0.0,
            )
            arr = raw.postprocess(**params)
        rgb = arr.astype(np.float32) / 65535.0
        capture = read_capture_info(path)
        meta.update({k: str(v) for k, v in capture.fields.items()})
        return LoadedImage(rgb=np.ascontiguousarray(rgb), path=path, is_raw=True,
                           meta=meta, capture=capture)

    # image deja developpee : on repasse en lineaire pour traiter correctement
    data = cv2.imread(path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if data is None:
        raise ValueError(f"format d'image non reconnu : {path}")
    if data.ndim == 2:
        data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
    if data.shape[2] == 4:
        data = data[..., :3]
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    maxv = 65535.0 if data.dtype == np.uint16 else 255.0
    enc = data.astype(np.float32) / maxv
    rgb = srgb_to_linear(enc)
    capture = read_capture_info(path)
    meta = {k: str(v) for k, v in capture.fields.items()}
    if half_size:
        rgb = cv2.resize(rgb, (rgb.shape[1] // 2, rgb.shape[0] // 2),
                         interpolation=cv2.INTER_AREA)
    return LoadedImage(rgb=np.ascontiguousarray(rgb), path=path, is_raw=False,
                       meta=meta, capture=capture)


def auto_exposure(rgb: np.ndarray, percentile: float = 99.7, target: float = 0.94,
                  max_ev: float = 3.5, min_ev: float = -1.0,
                  mid_target: float = 0.22) -> float:
    """Gain d'exposition, sous deux contraintes.

    Un RAW developpe en lineaire sans eclaircissement est souvent sombre (les
    fabricants gardent de la marge). On remonte donc, mais deux choses
    limitent la remontee :

    * les **hautes lumieres** : le haut de l'histogramme ne doit pas bruler ;
    * la **tonalite generale** : une scene nocturne ne doit pas se reveiller
      en plein jour sous pretexte qu'elle est sombre.

    On prend le plus petit des deux gains, et la contrainte de tonalite ne
    peut qu'abaisser le gain, jamais griser une scene claire (une photo de
    neige reste blanche).
    """
    sample = rgb[::4, ::4] if rgb.size > 4_000_000 else rgb
    high = float(np.percentile(sample, percentile))
    if high <= 1e-5:
        return 1.0
    gain_high = target / high
    median = float(np.median(sample))
    gain_mid = mid_target / median if median > 1e-6 else gain_high
    gain = min(gain_high, max(gain_mid, 1.0))
    lo = float(2.0 ** min_ev)
    hi = float(2.0 ** max_ev)
    return float(np.clip(gain, lo, hi))


def _dither8(x: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Bruit triangulaire +-0.5 LSB : supprime le banding sur 8 bits."""
    rng = rng or np.random.default_rng(12345)
    n = rng.random(x.shape, dtype=np.float32) - rng.random(x.shape, dtype=np.float32)
    return x + n / 255.0


def save_image(path: str, rgb_linear: np.ndarray, fmt: str = "tiff16",
               quality: int = 95, transfer: str = "srgb",
               metadata: Optional[Dict[str, Any]] = None,
               dither: bool = True,
               capture: Optional[CaptureInfo] = None,
               software: str = "naturaw") -> str:
    """Ecrit le resultat. ``fmt`` : tiff16, tiff8, png16, png, jpeg.

    ``capture`` porte les metadonnees du fichier d'origine : elles sont
    recopiees dans le TIFF (etiquettes standard) et dans le JPEG (bloc EXIF
    complet, date et position GPS comprises).
    """
    fmt = (fmt or "tiff16").lower()
    enc = apply_gamma(rgb_linear, transfer)
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    if fmt in ("tiff16", "tif16", "tiff", "tif"):
        import tifffile
        data = np.clip(enc * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
        # La description TIFF doit etre en ASCII 7 bits : un nom de fichier
        # accentue (ou un rapport en francais) faisait echouer l'ecriture.
        desc = json.dumps(metadata or {}, ensure_ascii=True)
        extra = tiff_extratags(capture, software) if capture else None
        tifffile.imwrite(path, data, photometric="rgb", compression="adobe_deflate",
                         description=desc, metadata=None, extratags=extra or ())
        return path

    if fmt in ("png16",):
        data = np.clip(enc * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
        _imwrite(path, cv2.cvtColor(data, cv2.COLOR_RGB2BGR), [])
        return path

    if fmt in ("png", "png8"):
        if dither:
            enc = _dither8(enc)
        data = np.clip(enc * 255.0 + 0.5, 0, 255).astype(np.uint8)
        _imwrite(path, cv2.cvtColor(data, cv2.COLOR_RGB2BGR),
                 [cv2.IMWRITE_PNG_COMPRESSION, 6])
        return path

    if fmt in ("jpeg", "jpg"):
        if dither:
            enc = _dither8(enc)
        data = np.clip(enc * 255.0 + 0.5, 0, 255).astype(np.uint8)
        params = [cv2.IMWRITE_JPEG_QUALITY, int(np.clip(quality, 1, 100)),
                  cv2.IMWRITE_JPEG_OPTIMIZE, 1]
        # chrominance pleine resolution (4:4:4) quand OpenCV le permet :
        # un sous-echantillonnage 4:2:0 annulerait une partie du travail fait
        # sur la couleur
        factor = getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR", None)
        f444 = getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_444", None)
        if factor is not None and f444 is not None:
            params += [factor, f444]
        _imwrite(path, cv2.cvtColor(data, cv2.COLOR_RGB2BGR), params)
        if capture:
            attach_to_jpeg(path, exif_payload(capture, software))
        return path

    if fmt in ("tiff8", "tif8"):
        import tifffile
        if dither:
            enc = _dither8(enc)
        data = np.clip(enc * 255.0 + 0.5, 0, 255).astype(np.uint8)
        extra = tiff_extratags(capture, software) if capture else None
        tifffile.imwrite(path, data, photometric="rgb", compression="adobe_deflate",
                         description=json.dumps(metadata or {}, ensure_ascii=True),
                         extratags=extra or ())
        return path

    raise ValueError(f"format de sortie inconnu : {fmt}")


def _imwrite(path: str, bgr: np.ndarray, params) -> None:
    """Ecriture robuste aux chemins non ASCII (via imencode)."""
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, bgr, params)
    if not ok:
        raise IOError(f"echec de l'encodage {ext} pour {path}")
    with open(path, "wb") as fh:
        fh.write(buf.tobytes())


def default_output_name(path: str, suffix: str = "_naturaw", fmt: str = "tiff16") -> str:
    base, _ = os.path.splitext(os.path.basename(path))
    ext = {"tiff16": ".tif", "tiff8": ".tif", "tiff": ".tif", "tif": ".tif",
           "png16": ".png", "png": ".png", "png8": ".png",
           "jpeg": ".jpg", "jpg": ".jpg"}.get(fmt.lower(), ".tif")
    return f"{base}{suffix}{ext}"
