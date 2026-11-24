"""Recuperation des ombres et des hautes lumieres, sans halo.

C'est ce qui sauve vraiment une photo "inexploitable" : trois diaphragmes de
sous-exposition se rattrapent, a condition de ne pas se contenter d'un gain
global (qui brule le ciel) ni d'une courbe en S (qui ecrase le contraste
local).

Methode : on separe la luminance en une **base** (ce que l'oeil percoit comme
l'eclairement de la scene) et un **detail** (la matiere locale), dans le
domaine logarithmique ou le contraste est multiplicatif. Seule la base est
modifiee ; le detail est reporte tel quel. La base est calculee avec un filtre
guide, qui suit les contours : c'est ce qui evite le halo clair autour des
objets sombres, defaut classique des outils de recuperation d'ombres.

Le gain est applique en *rapport* sur les trois canaux : la teinte et la
saturation relatives ne bougent pas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .colorspace import luminance
from .filters import guided_filter, resize_long_side

_EPS = 1e-4


@dataclass
class ToneParams:
    """Reglages de la recuperation tonale."""

    shadows: float = 0.0
    """Ouverture des ombres, 0 a 1 (1 ~ trois diaphragmes dans les noirs)."""

    highlights: float = 0.0
    """Recuperation des hautes lumieres, 0 a 1."""

    auto: bool = True
    """Decider automatiquement de l'ouverture des ombres selon l'histogramme."""

    auto_max: float = 0.8
    """Ouverture maximale que le mode automatique s'autorise."""

    local_contrast: float = 0.0
    """Renforcement du contraste local (0 = aucun)."""

    radius: float = 0.04
    """Rayon de la base, en fraction du grand cote."""


def auto_shadow_amount(rgb: np.ndarray, target: float = 0.12,
                       limit: float = 0.8) -> float:
    """Combien ouvrir les ombres, d'apres la repartition des tons.

    On compare la mediane des tons sombres a une cible ; une photo deja
    correctement exposee renvoie 0.
    """
    sample = rgb[::4, ::4] if rgb.size > 4_000_000 else rgb
    lum = luminance(sample)
    low = float(np.percentile(lum, 25.0))
    if low <= 0.0:
        low = float(np.percentile(lum, 50.0))
    if low >= target or low <= 0.0:
        return 0.0
    # chaque diaphragme manquant vaut un tiers d'ouverture
    stops = float(np.log2(target / max(low, 1e-5)))
    return float(np.clip(stops / 3.0, 0.0, limit))


def planned_gain(rgb: np.ndarray, params: ToneParams = ToneParams()) -> Optional[np.ndarray]:
    """Gain de luminance que la recuperation tonale appliquera, par pixel.

    Sert a prevenir le debruiteur : une ombre qui sera remontee de trois
    diaphragmes doit etre nettoyee comme si son bruit etait trois fois plus
    grand, sinon on ne fait que l'amplifier proprement.
    """
    shadows = float(params.shadows)
    if params.auto and shadows <= 0.0:
        shadows = auto_shadow_amount(rgb, limit=params.auto_max)
    if shadows <= 1e-3:
        return None
    rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    small, scale = resize_long_side(rgb, 900)
    lum = np.maximum(luminance(small), _EPS)
    base = guided_filter(np.log2(lum), np.log2(lum),
                         max(4, int(round(params.radius * max(small.shape[:2])))), 0.2)
    darkness = np.clip((-2.0 - base) / 6.0 + 0.5, 0.0, 1.0)
    darkness *= darkness
    gain = np.exp2((3.0 * shadows) * darkness)
    if scale < 1.0:
        gain = cv2.resize(gain, (rgb.shape[1], rgb.shape[0]),
                          interpolation=cv2.INTER_LINEAR)
    return gain.astype(np.float32, copy=False)


def tone_rescue(rgb: np.ndarray, params: ToneParams = ToneParams(),
                report: Optional[dict] = None) -> np.ndarray:
    """Ouvre les ombres et retient les hautes lumieres d'une image lineaire."""
    rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    shadows = float(params.shadows)
    if params.auto and shadows <= 0.0:
        shadows = auto_shadow_amount(rgb, limit=params.auto_max)
    highlights = float(np.clip(params.highlights, 0.0, 1.0))
    local = float(np.clip(params.local_contrast, 0.0, 1.0))
    if shadows <= 1e-3 and highlights <= 1e-3 and local <= 1e-3:
        if report is not None:
            report["tonalite"] = "inchangee"
        return rgb

    lum = luminance(rgb)
    np.maximum(lum, _EPS, out=lum)
    log_lum = np.log2(lum)

    radius = max(4, int(round(params.radius * max(rgb.shape[0], rgb.shape[1]))))
    # eps en (stops)^2 : en dessous de ~0,45 stop d'ecart, le filtre lisse
    base = guided_filter(log_lum, log_lum, radius, 0.2)
    detail = log_lum - base

    out_base = base
    if shadows > 0.0:
        # poids maximal dans les noirs, nul au-dessus du gris moyen
        darkness = np.clip((-2.0 - base) / 6.0 + 0.5, 0.0, 1.0)
        darkness *= darkness
        out_base = out_base + (3.0 * shadows) * darkness
    if highlights > 0.0:
        brightness = np.clip((base + 1.5) / 2.5, 0.0, 1.0)
        brightness *= brightness
        out_base = out_base - (2.0 * highlights) * brightness

    out_log = out_base + detail * (1.0 + 0.6 * local)
    new_lum = np.exp2(out_log)
    gain = new_lum / lum
    np.clip(gain, 0.05, 64.0, out=gain)
    out = rgb * gain[..., None]
    np.clip(out, 0.0, None, out=out)
    if report is not None:
        report["tonalite"] = (f"ombres +{shadows * 3.0:.1f} EV"
                              + (f", hautes lumieres -{highlights * 2.0:.1f} EV" if highlights else "")
                              + (f", contraste local +{local:.1f}" if local else ""))
    return out
