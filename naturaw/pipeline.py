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
from .noise import quick_sigma
from .rescue import RescueParams, apply_rescue
from .sharpen import SharpenParams, sharpen_linear_rgb
from .tone import ToneParams, planned_gain, tone_rescue
from . import __version__
from .rawio import (IMAGE_EXTENSIONS, RAW_EXTENSIONS, auto_exposure,
                    default_output_name, load_image, save_image)

ProgressFn = Callable[[str, float], None]


@dataclass
class Settings:
    """Tous les reglages d'un traitement."""

    rescue: RescueParams = field(default_factory=RescueParams)
    denoise: DenoiseParams = field(default_factory=DenoiseParams)
    deblur: DeblurParams = field(default_factory=DeblurParams)
    sharpen: SharpenParams = field(default_factory=SharpenParams)
    tone: ToneParams = field(default_factory=ToneParams)
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

    # --- sauvetage : pixels chauds, bandes de lecture ---------------------
    if progress:
        progress("sauvetage", 0.12)
    t0 = time.time()
    sigma0 = quick_sigma(rgb)
    report["bruit_rapide"] = round(sigma0, 5)
    rgb = apply_rescue(rgb, settings.rescue, sigma0, report=report)
    report["temps_sauvetage"] = round(time.time() - t0, 2)

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
        # Si les ombres vont etre remontees, le debruiteur doit le savoir :
        # sinon il nettoie un bruit juge faible, que l'on amplifie ensuite.
        visibility = planned_gain(rgb, settings.tone)
        if visibility is not None and report is not None:
            report["visibilite"] = (f"debruitage renforce jusqu'a x{float(visibility.max()):.1f} "
                                    "dans les ombres a remonter")
        rgb = denoise_linear_rgb(rgb, settings.denoise, report=report,
                                 visibility=visibility)
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

    # --- nettete de capture -----------------------------------------------
    if progress:
        progress("nettete", 0.68)
    t0 = time.time()
    sharpen = settings.sharpen
    if str(report.get("deblur", "")).startswith("noyau"):
        # une deconvolution vient d'avoir lieu : on n'en rajoute qu'un peu
        sharpen = SharpenParams(**{**sharpen.__dict__,
                                   "amount": sharpen.amount * 0.5})
    rgb = sharpen_linear_rgb(rgb, sharpen, noise_sigma=sigma, report=report)
    report["temps_nettete"] = round(time.time() - t0, 2)

    # --- tonalite : ouverture des ombres ----------------------------------
    if progress:
        progress("tonalite", 0.72)
    t0 = time.time()
    rgb = tone_rescue(rgb, settings.tone, report=report)
    report["temps_tonalite"] = round(time.time() - t0, 2)

    # --- redressement -----------------------------------------------------
    if progress:
        progress("redressement", 0.8)
    t0 = time.time()
    rgb = deskew_image(rgb, settings.deskew, report=report)
    report["temps_redressement"] = round(time.time() - t0, 2)

    np.clip(rgb, 0.0, None, out=rgb)
    return rgb


def preview_file(path: str, out_path: str, settings: Settings,
                 long_side: int = 1800, crop: Sequence[int] = (1200, 800)
                 ) -> Dict[str, Any]:
    """Ecrit un apercu double : l'image entiere reduite, et une zone a 100 %.

    Le second fichier porte le suffixe ``-100`` ; c'est celui sur lequel on
    juge le bruit et la nettete, puisqu'il est traite exactement comme le
    fichier final.
    """
    report: Dict[str, Any] = {"entree": path, "sortie": out_path}
    start = time.time()
    loaded = load_image(path, half_size=False, demosaic=settings.demosaic,
                        camera_wb=settings.camera_wb, color_space=settings.color_space,
                        highlight_mode=settings.highlight)
    report["dimensions"] = f"{loaded.rgb.shape[1]}x{loaded.rgb.shape[0]}"
    report["megapixels"] = round(loaded.rgb.shape[0] * loaded.rgb.shape[1] / 1e6, 1)
    whole, detail = process_preview(loaded.rgb, settings, long_side=long_side,
                                    crop=crop, report=report)
    base, ext = os.path.splitext(out_path)
    zoom_path = f"{base}-100{ext}"
    for target, img in ((out_path, whole), (zoom_path, detail)):
        save_image(target, img, fmt=settings.fmt, quality=settings.quality,
                   transfer=settings.transfer, dither=settings.dither,
                   metadata={"logiciel": "naturaw " + __version__, "apercu": True},
                   capture=loaded.capture, software=f"naturaw {__version__}")
    report["sortie_100"] = zoom_path
    report["dimensions_sortie"] = f"{whole.shape[1]}x{whole.shape[0]}"
    report["duree"] = round(time.time() - start, 2)
    return report


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
    meta = {"logiciel": f"naturaw {__version__}", "source": os.path.basename(path),
            "traitement": {k: v for k, v in report.items() if isinstance(v, (str, int, float))}}
    if loaded.capture:
        report["prise_de_vue"] = loaded.capture.summary()
    save_image(out_path, out, fmt=settings.fmt, quality=settings.quality,
               transfer=settings.transfer, metadata=meta, dither=settings.dither,
               capture=loaded.capture, software=f"naturaw {__version__}")
    report["duree"] = round(time.time() - start, 2)
    if progress:
        progress("termine", 1.0)
    return report


def most_detailed_region(rgb: np.ndarray, size: Sequence[int] = (1200, 800)):
    """Fenetre la plus texturee de l'image, en coordonnees (x0, y0, x1, y1).

    C'est la zone qu'un photographe regarde pour juger le bruit et la
    nettete : celle qui porte de la matiere.
    """
    import cv2
    h, w = rgb.shape[:2]
    cw, ch = int(min(size[0], w)), int(min(size[1], h))
    if cw >= w and ch >= h:
        return 0, 0, w, h
    from .colorspace import luminance
    scale = min(1.0, 600.0 / max(h, w))
    small = cv2.resize(luminance(rgb), (max(8, int(w * scale)), max(8, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    grad = cv2.magnitude(cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3),
                         cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3))
    k = (max(3, int(cw * scale) | 1), max(3, int(ch * scale) | 1))
    energy = cv2.boxFilter(grad, -1, k, normalize=True, borderType=cv2.BORDER_REFLECT)
    yy, xx = np.unravel_index(int(np.argmax(energy)), energy.shape)
    cx = int(np.clip(xx / scale, cw // 2, w - cw // 2))
    cy = int(np.clip(yy / scale, ch // 2, h - ch // 2))
    return cx - cw // 2, cy - ch // 2, cx - cw // 2 + cw, cy - ch // 2 + ch


def process_preview(rgb: np.ndarray, settings: Settings,
                    long_side: int = 1800, crop: Sequence[int] = (1200, 800),
                    report: Optional[Dict[str, Any]] = None):
    """Apercu double : l'image entiere reduite, et une fenetre a 100 %.

    Juger un debruitage sur une image reduite n'a pas de sens -- la reduction
    cache a la fois le bruit et le detail. La fenetre a 100 %, elle, est
    traitee exactement comme le fichier final : ce que l'on y voit est ce que
    l'on aura.
    """
    report = report if report is not None else {}
    x0, y0, x1, y1 = most_detailed_region(rgb, crop)
    detail_settings = Settings(**{**settings.__dict__, "max_side": 0})
    detail = process_array(np.ascontiguousarray(rgb[y0:y1, x0:x1]), detail_settings,
                           report=report)
    whole_settings = Settings(**{**settings.__dict__,
                                 "max_side": settings.max_side or long_side})
    whole = process_array(rgb, whole_settings, report={})
    report["apercu_zone"] = f"{x1 - x0}x{y1 - y0} a partir de ({x0}, {y0})"
    return whole, detail


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


def auto_jobs(paths: Sequence[str] = (), limit: int = 8) -> int:
    """Nombre de photos a traiter de front, choisi tout seul.

    Deux contraintes : les coeurs disponibles et la memoire. Mesure sur ce
    moteur : une photo en cours de traitement occupe environ 60 Mo par
    megapixel, tout compris. La taille est estimee sur le premier fichier du
    lot (un RAW compresse pese environ 1,2 octet par pixel).

    Une photo par coeur est le reglage le plus rapide (mesure : six photos de
    8,6 Mpx en 23,8 s a un seul processus, 10,9 s a quatre sur quatre coeurs),
    parce que le traitement d'une seule image ne sature pas les coeurs.
    """
    cpus = os.cpu_count() or 2
    jobs = max(1, min(int(limit), cpus))
    try:
        free = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return jobs
    megapixels = 24.0
    for p in paths[:1]:
        try:
            megapixels = max(1.0, os.path.getsize(p) / 1.2e6)
        except OSError:
            pass
    need = megapixels * 70e6                  # ~60 Mo/Mpx mesures, avec marge
    if need > 0:
        jobs = max(1, min(jobs, int(free * 0.7 // need)))
    return jobs


def _pool_context():
    """Contexte multiprocessing sur lequel on peut compter.

    Le mode ``fork`` de Linux duplique un processus qui a deja des fils
    d'execution (OpenCV, OpenMP de LibRaw) : les verrous sont copies dans un
    etat indetermine et l'enfant se bloque pour toujours -- rawpy previent
    lui-meme du probleme. On demarre donc les processus proprement.
    """
    import multiprocessing as mp
    for name in ("forkserver", "spawn"):
        if name not in mp.get_all_start_methods():
            continue
        try:
            ctx = mp.get_context(name)
        except (ValueError, RuntimeError):
            continue
        if name == "forkserver":
            # Par defaut le serveur precharge ``__main__`` : un script
            # utilisateur sans garde ``if __name__ == "__main__"`` serait
            # alors reexecute dans chaque processus de travail. On precharge
            # ce module-ci a la place, qui suffit.
            try:
                ctx.set_forkserver_preload(["naturaw.pipeline"])
            except Exception:
                pass
        return ctx
    return None


def _worker(args):
    path, out_path, settings, threads, preview = args
    try:
        import cv2
        cv2.setNumThreads(max(1, int(threads)))
    except Exception:
        pass
    try:
        if preview:
            return preview_file(path, out_path, settings)
        return process_file(path, out_path, settings)
    except Exception as exc:  # on ne perd pas le lot pour un fichier
        return {"entree": path, "sortie": out_path, "erreur": f"{type(exc).__name__}: {exc}"}


def process_many(paths: Sequence[str], out_dir: str, settings: Settings,
                 jobs: int = 1, progress: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
                 preview: bool = False) -> List[Dict[str, Any]]:
    """Traite un lot de fichiers (eventuellement en parallele).

    ``preview`` ecrit des apercus (image entiere reduite + zone a 100 %) au
    lieu du fichier final.
    """
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
        tasks.append((p, out_path, settings, max(1, (os.cpu_count() or 2) // jobs),
                      bool(preview)))

    reports: List[Dict[str, Any]] = []
    if jobs == 1:
        for i, t in enumerate(tasks):
            rep = _worker(t)
            reports.append(rep)
            if progress:
                progress(i + 1, len(tasks), rep)
        return reports

    from concurrent.futures import ProcessPoolExecutor
    ctx = _pool_context()
    try:
        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as pool:
            for i, rep in enumerate(pool.map(_worker, tasks)):
                reports.append(rep)
                if progress:
                    progress(i + 1, len(tasks), rep)
        return reports
    except Exception as exc:       # pas de processus disponible : on continue seul
        reports = []
        for i, task in enumerate(tasks):
            rep = _worker(task)
            if i == 0:
                rep["avertissement"] = (f"traitement parallele indisponible ({exc}), "
                                        "repli sur un seul processus")
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
    if report.get("prise_de_vue"):
        lines.append(f"   prise de vue      : {report['prise_de_vue']}")
    if report.get("noise_sigma"):
        lines.append(f"   bruit mesure      : {report['noise_sigma']}")
    if report.get("pixels_chauds"):
        lines.append(f"   pixels chauds     : {report['pixels_chauds']}")
    if report.get("bandes") and report["bandes"] != "aucune bande detectee":
        lines.append(f"   bandes            : {report['bandes']}")
    if report.get("debruitage"):
        lines.append(f"   debruitage        : {report['debruitage']}")
    if report.get("tonalite") and report["tonalite"] != "inchangee":
        lines.append(f"   tonalite          : {report['tonalite']}")
    if report.get("exposition"):
        lines.append(f"   exposition        : {report['exposition']}")
    if report.get("deblur"):
        lines.append(f"   deflou            : {report['deblur']}")
    if report.get("nettete"):
        lines.append(f"   nettete           : {report['nettete']}")
    if report.get("deskew"):
        lines.append(f"   redressement      : {report['deskew']}")
    lines.append(f"   sortie            : {report.get('sortie', '?')}"
                 f"  ({report.get('dimensions_sortie', '?')}, {report.get('duree', '?')} s)")
    if report.get("sortie_100"):
        lines.append(f"   zone a 100 %      : {report['sortie_100']}"
                     f"  ({report.get('apercu_zone', '?')})")
    return "\n".join(lines)
