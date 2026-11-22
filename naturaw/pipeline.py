"""Chaine de traitement complete, du RAW au fichier final.

Ordre des operations, et pourquoi :

1. **Developpement** du RAW en lumiere lineaire (aucun lissage du decodeur).
2. **Exposition** : gain doux et borne, avant tout le reste, pour que le
   modele de bruit soit mesure sur les valeurs reellement ecrites.
3. **Debruitage** : avant la deconvolution, qui amplifierait le bruit.
4. **Deflou de bouge** : sur l'image debruitee (l'estimation du noyau est
   beaucoup plus fiable sans bruit) et *avant* toute rotation, car la
   convolution n'est valable que dans la geometrie d'origine.
5. **Redressement** : en dernier, une seule interpolation geometrique
   (Lanczos) pour ne degrader la nettete qu'une fois.
6. **Courbe de transfert** et ecriture.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .deblur import DeblurParams, deblur_linear_rgb
from .denoise import DenoiseParams, PRESETS, denoise_linear_rgb
from .deskew import DeskewParams, deskew_image
from .rawio import (IMAGE_EXTENSIONS, RAW_EXTENSIONS, auto_exposure,
                    default_output_name, load_image, save_image)

ProgressFn = Callable[[str, float], None]


@dataclass
class Settings:
    """Tous les reglages d'un traitement."""

    denoise: DenoiseParams = field(default_factory=DenoiseParams)
    deblur: DeblurParams = field(default_factory=DeblurParams)
    deskew: DeskewParams = field(default_factory=DeskewParams)

    # developpement
    demosaic: str = "ahd"
    camera_wb: bool = True
    color_space: str = "srgb"
    highlight: int = 0
    half_size: bool = False
    max_side: int = 0

    # sortie
    fmt: str = "tiff16"
    quality: int = 95
    transfer: str = "srgb"
    exposure: str = "auto"
    dither: bool = True
    suffix: str = "_naturaw"
    overwrite: bool = False

    @staticmethod
    def denoise_preset(name: str) -> float:
        key = (name or "normal").lower()
        if key in PRESETS:
            return PRESETS[key]
        return float(key)


def process_array(rgb: np.ndarray, settings: Settings,
                  report: Optional[Dict[str, Any]] = None,
                  progress: Optional[ProgressFn] = None) -> np.ndarray:
    """Applique exposition, debruitage, deflou et redressement a une image lineaire."""
    report = report if report is not None else {}
    rgb = np.ascontiguousarray(rgb, dtype=np.float32)

    if settings.max_side and max(rgb.shape[:2]) > settings.max_side:
        import cv2
        scale = settings.max_side / float(max(rgb.shape[:2]))
        rgb = cv2.resize(rgb, (max(1, int(round(rgb.shape[1] * scale))),
                               max(1, int(round(rgb.shape[0] * scale)))),
                         interpolation=cv2.INTER_AREA)
        report["reduction"] = f"ramenee a {rgb.shape[1]}x{rgb.shape[0]}"

    # --- exposition -------------------------------------------------------
    mode = str(settings.exposure or "auto").lower()
    if mode in ("auto", "oui", "yes"):
        gain = auto_exposure(rgb)
    elif mode in ("none", "non", "off", "0"):
        gain = 1.0
    else:
        gain = float(2.0 ** float(mode))
    if abs(gain - 1.0) > 1e-3:
        rgb = rgb * np.float32(gain)
        report["exposition"] = f"{np.log2(gain):+.2f} EV"
    else:
        report["exposition"] = "inchangee"

    # --- debruitage -------------------------------------------------------
    if progress:
        progress("debruitage", 0.25)
    t0 = time.time()
    if settings.denoise.strength > 0:
        rgb = denoise_linear_rgb(rgb, settings.denoise, report=report)
        report["temps_debruitage"] = round(time.time() - t0, 2)
    else:
        report["noise_luma_sigma"] = 0.0
        report["debruitage"] = "desactive"
    sigma = float(report.get("noise_luma_sigma", 0.002) or 0.002)

    # --- deflou de bouge --------------------------------------------------
    if progress:
        progress("deflou", 0.55)
    t0 = time.time()
    rgb = deblur_linear_rgb(rgb, settings.deblur, noise_sigma=sigma, report=report)
    report["temps_deflou"] = round(time.time() - t0, 2)

    # --- redressement -----------------------------------------------------
    if progress:
        progress("redressement", 0.8)
    t0 = time.time()
    rgb = deskew_image(rgb, settings.deskew, report=report)
    report["temps_redressement"] = round(time.time() - t0, 2)

    np.clip(rgb, 0.0, None, out=rgb)
    return rgb


def process_file(path: str, out_path: str, settings: Settings,
                 progress: Optional[ProgressFn] = None) -> Dict[str, Any]:
    """Traite un fichier et ecrit le resultat. Renvoie un rapport."""
    report: Dict[str, Any] = {"entree": path, "sortie": out_path}
    start = time.time()
    if progress:
        progress("lecture", 0.05)
    loaded = load_image(path, half_size=settings.half_size, demosaic=settings.demosaic,
                        camera_wb=settings.camera_wb, color_space=settings.color_space,
                        highlight_mode=settings.highlight)
    report["raw"] = loaded.is_raw
    report["dimensions"] = f"{loaded.rgb.shape[1]}x{loaded.rgb.shape[0]}"
    report["megapixels"] = round(loaded.rgb.shape[0] * loaded.rgb.shape[1] / 1e6, 1)
    for k, v in loaded.meta.items():
        report.setdefault(f"exif_{k}", v)

    out = process_array(loaded.rgb, settings, report=report, progress=progress)
    report["dimensions_sortie"] = f"{out.shape[1]}x{out.shape[0]}"

    if progress:
        progress("ecriture", 0.95)
    meta = {"logiciel": "naturaw 0.1.0", "source": os.path.basename(path),
            "traitement": {k: v for k, v in report.items() if isinstance(v, (str, int, float))}}
    save_image(out_path, out, fmt=settings.fmt, quality=settings.quality,
               transfer=settings.transfer, metadata=meta, dither=settings.dither)
    report["duree"] = round(time.time() - start, 2)
    if progress:
        progress("termine", 1.0)
    return report


def collect_inputs(paths: Sequence[str], recursive: bool = False,
                   raw_only: bool = False) -> List[str]:
    """Developpe une liste de fichiers et/ou dossiers en liste de fichiers."""
    exts = set(RAW_EXTENSIONS) if raw_only else set(RAW_EXTENSIONS) | set(IMAGE_EXTENSIONS)
    out: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            if recursive:
                for root, _dirs, files in os.walk(p):
                    for f in sorted(files):
                        if os.path.splitext(f)[1].lower() in exts:
                            out.append(os.path.join(root, f))
            else:
                for f in sorted(os.listdir(p)):
                    full = os.path.join(p, f)
                    if os.path.isfile(full) and os.path.splitext(f)[1].lower() in exts:
                        out.append(full)
        elif os.path.isfile(p):
            out.append(p)
    # dedoublonnage en gardant l'ordre
    seen = set()
    uniq = []
    for p in out:
        key = os.path.abspath(p)
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def _worker(args):
    path, out_path, settings, threads = args
    try:
        import cv2
        cv2.setNumThreads(max(1, int(threads)))
    except Exception:
        pass
    try:
        return process_file(path, out_path, settings)
    except Exception as exc:  # on ne perd pas le lot pour un fichier
        return {"entree": path, "sortie": out_path, "erreur": f"{type(exc).__name__}: {exc}"}


def process_many(paths: Sequence[str], out_dir: str, settings: Settings,
                 jobs: int = 1, progress: Optional[Callable[[int, int, Dict[str, Any]], None]] = None
                 ) -> List[Dict[str, Any]]:
    """Traite un lot de fichiers (eventuellement en parallele)."""
    jobs = max(1, int(jobs))
    tasks = []
    for p in paths:
        name = default_output_name(p, settings.suffix, settings.fmt)
        out_path = os.path.join(out_dir, name) if out_dir else os.path.join(
            os.path.dirname(os.path.abspath(p)), name)
        if os.path.exists(out_path) and not settings.overwrite:
            base, ext = os.path.splitext(out_path)
            n = 2
            while os.path.exists(f"{base}-{n}{ext}"):
                n += 1
            out_path = f"{base}-{n}{ext}"
        tasks.append((p, out_path, settings, max(1, (os.cpu_count() or 2) // jobs)))

    reports: List[Dict[str, Any]] = []
    if jobs == 1:
        for i, t in enumerate(tasks):
            rep = _worker(t)
            reports.append(rep)
            if progress:
                progress(i + 1, len(tasks), rep)
        return reports

    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for i, rep in enumerate(pool.map(_worker, tasks)):
            reports.append(rep)
            if progress:
                progress(i + 1, len(tasks), rep)
    return reports


def summarize(report: Dict[str, Any]) -> str:
    """Resume lisible d'un rapport de traitement."""
    if "erreur" in report:
        return f"ECHEC  {os.path.basename(report.get('entree', '?'))} : {report['erreur']}"
    lines = [f"{os.path.basename(report.get('entree', '?'))}  "
             f"({report.get('dimensions', '?')}, {report.get('megapixels', '?')} Mpx)"]
    if report.get("noise_sigma"):
        lines.append(f"   bruit mesure      : {report['noise_sigma']}")
    if report.get("debruitage"):
        lines.append(f"   debruitage        : {report['debruitage']}")
    if report.get("exposition"):
        lines.append(f"   exposition        : {report['exposition']}")
    if report.get("deblur"):
        lines.append(f"   deflou            : {report['deblur']}")
    if report.get("deskew"):
        lines.append(f"   redressement      : {report['deskew']}")
    lines.append(f"   sortie            : {report.get('sortie', '?')}"
                 f"  ({report.get('dimensions_sortie', '?')}, {report.get('duree', '?')} s)")
    return "\n".join(lines)
