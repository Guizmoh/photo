"""Redressement automatique des photos legerement penchees.

L'idee : dans une scene construite (architecture, horizon, murs, poteaux,
cadres, tables...) les lignes dominantes sont horizontales ou verticales. Si
l'appareil etait penche d'un angle a, toutes ces lignes sont inclinees du meme
angle. On detecte donc les segments de droite, on ramene chaque orientation
*modulo 90 degres* (une verticale penchee de 2 deg et une horizontale penchee
de 2 deg portent la meme information), et on cherche le pic de cette
distribution, pondere par la longueur des segments au carre : une longue
droite est une mesure beaucoup plus fiable qu'un petit segment.

Un indice de confiance accompagne l'angle : s'il est faible (scene sans
structure, feuillage, portrait en gros plan), on ne touche pas a l'image.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .colorspace import linear_to_srgb, luminance
from .filters import resize_long_side


@dataclass
class DeskewParams:
    """Reglages du redressement."""

    enabled: bool = True
    max_angle: float = 8.0
    """Correction maximale acceptee (degres). Au-dela, c'est un cadrage voulu."""

    min_angle: float = 0.08
    """En dessous, on ne touche pas a l'image (evite un reechantillonnage inutile)."""

    min_confidence: float = 0.22
    """Confiance minimale (part des segments d'accord avec l'angle retenu)."""

    crop: str = "aspect"
    """``aspect`` (plus grand cadre de meme proportion, defaut), ``max``
    (plus grande aire, les proportions changent) ou ``none`` (bords vides
    remplis par etirement du bord)."""

    analysis_long_side: int = 1600
    """Resolution d'analyse des lignes."""

    angle: Optional[float] = None
    """Angle impose (degres, sens trigonometrique) ; None = automatique."""


@dataclass
class TiltEstimate:
    angle: float = 0.0
    confidence: float = 0.0
    segments: int = 0
    weight: float = 0.0
    reason: str = ""

    def describe(self) -> str:
        return (f"{self.angle:+.2f} deg (confiance {self.confidence:.2f}, "
                f"{self.segments} segments)")


def _detect_segments(gray8: np.ndarray):
    """Segments de droite : LSD si disponible, sinon Hough probabiliste."""
    try:
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
        lines = lsd.detect(gray8)[0]
        if lines is not None and len(lines):
            return lines.reshape(-1, 4)
    except Exception:
        pass
    edges = cv2.Canny(gray8, 60, 180, apertureSize=3)
    minlen = max(30, int(0.04 * max(gray8.shape)))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720.0, threshold=60,
                            minLineLength=minlen, maxLineGap=4)
    if lines is None:
        return np.zeros((0, 4), np.float32)
    return lines.reshape(-1, 4).astype(np.float32)


def estimate_tilt(rgb_or_luma: np.ndarray,
                  params: DeskewParams = DeskewParams()) -> TiltEstimate:
    """Estime l'inclinaison de l'image (degres a appliquer pour la redresser)."""
    img = rgb_or_luma
    luma = luminance(img) if img.ndim == 3 else img
    # detection sur une image encodee en gamma : les contours des zones
    # sombres comptent autant que ceux des hautes lumieres
    small, _ = resize_long_side(linear_to_srgb(np.clip(luma, 0, 1)),
                                params.analysis_long_side)
    gray8 = np.clip(small * 255.0, 0, 255).astype(np.uint8)
    segs = _detect_segments(gray8)
    if len(segs) == 0:
        return TiltEstimate(reason="aucun segment detecte")

    x1, y1, x2, y2 = segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3]
    dx, dy = x2 - x1, y2 - y1
    length = np.sqrt(dx * dx + dy * dy)
    keep = length >= max(12.0, 0.015 * max(gray8.shape))
    if not np.any(keep):
        return TiltEstimate(reason="aucun segment assez long")
    dx, dy, length = dx[keep], dy[keep], length[keep]

    # orientation ramenee dans ]-45, 45] : une verticale penchee et une
    # horizontale penchee du meme angle donnent la meme valeur
    ang = np.degrees(np.arctan2(dy, dx))
    dev = np.mod(ang + 45.0, 90.0) - 45.0
    weight = length ** 2
    sel = np.abs(dev) <= params.max_angle
    if not np.any(sel):
        return TiltEstimate(segments=int(len(dev)), reason="aucune ligne proche de l'horizontale ou de la verticale")
    dev_sel, w_sel = dev[sel], weight[sel]

    bin_w = 0.1
    nb = int(round(2 * params.max_angle / bin_w)) + 1
    idx = np.clip(((dev_sel + params.max_angle) / bin_w).astype(int), 0, nb - 1)
    hist = np.bincount(idx, weights=w_sel, minlength=nb).astype(np.float64)
    # lissage : les mesures ont une incertitude de l'ordre de 0.2-0.3 deg
    hist = cv2.GaussianBlur(hist.reshape(-1, 1).astype(np.float32), (1, 9), 0).ravel()
    peak = int(np.argmax(hist))
    centre = peak * bin_w - params.max_angle
    # affinage : moyenne ponderee autour du pic
    near = np.abs(dev_sel - centre) <= 0.8
    if np.any(near):
        angle = float(np.sum(dev_sel[near] * w_sel[near]) / np.sum(w_sel[near]))
    else:
        angle = float(centre)
    total = float(weight.sum())
    conf = float(np.sum(w_sel[near]) / total) if total > 0 else 0.0
    return TiltEstimate(angle=-angle, confidence=conf, segments=int(np.count_nonzero(near)),
                        weight=total, reason="ok")


def largest_inner_rect_same_aspect(w: float, h: float, angle_deg: float) -> Tuple[float, float]:
    """Plus grand rectangle *de meme proportion* inscrit dans l'image tournee.

    Un cadre centre (W', H') tient dans un rectangle (w, h) tourne de a si
    ``W'cos a + H'sin a <= w`` et ``W'sin a + H'cos a <= h``. En imposant
    W'/H' = w/h, l'echelle maximale est le minimum des deux contraintes.
    C'est ce que l'on veut d'une photo : garder son format.
    """
    if w <= 0 or h <= 0:
        return 0.0, 0.0
    ang = np.deg2rad(abs(angle_deg) % 180.0)
    if ang > np.pi / 2:
        ang = np.pi - ang
    sin_a, cos_a = abs(np.sin(ang)), abs(np.cos(ang))
    s1 = w / max(w * cos_a + h * sin_a, 1e-9)
    s2 = h / max(w * sin_a + h * cos_a, 1e-9)
    s = min(s1, s2)
    return float(max(w * s, 1.0)), float(max(h * s, 1.0))


def largest_inner_rect(w: float, h: float, angle_deg: float) -> Tuple[float, float]:
    """Plus grand rectangle inscrit *en aire* (les proportions changent)."""
    if w <= 0 or h <= 0:
        return 0.0, 0.0
    ang = np.deg2rad(abs(angle_deg) % 180.0)
    if ang > np.pi / 2:
        ang = np.pi - ang
    sin_a, cos_a = abs(np.sin(ang)), abs(np.cos(ang))
    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)
    if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr = (w * cos_a - h * sin_a) / cos_2a
        hr = (h * cos_a - w * sin_a) / cos_2a
    return float(max(wr, 1.0)), float(max(hr, 1.0))


def rotate_image(img: np.ndarray, angle_deg: float, crop: str = "aspect",
                 interpolation: int = cv2.INTER_LANCZOS4) -> np.ndarray:
    """Tourne l'image (interpolation Lanczos) et recadre selon ``crop``."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0 - 0.5, h / 2.0 - 0.5), -angle_deg, 1.0)
    out = cv2.warpAffine(img, m, (w, h), flags=interpolation,
                         borderMode=cv2.BORDER_REPLICATE)
    mode = "none" if crop is False else ("aspect" if crop is True else str(crop).lower())
    if mode in ("none", "non", "off"):
        return out
    if mode == "max":
        wr, hr = largest_inner_rect(w, h, angle_deg)
    else:
        wr, hr = largest_inner_rect_same_aspect(w, h, angle_deg)
    x0 = int(round((w - wr) / 2.0))
    y0 = int(round((h - hr) / 2.0))
    x1 = min(w, x0 + int(round(wr)))
    y1 = min(h, y0 + int(round(hr)))
    return np.ascontiguousarray(out[max(0, y0):y1, max(0, x0):x1])


def deskew_image(rgb: np.ndarray, params: DeskewParams = DeskewParams(),
                 report: Optional[dict] = None,
                 estimate: Optional[TiltEstimate] = None) -> np.ndarray:
    """Redresse une image si (et seulement si) l'inclinaison est credible."""
    if not params.enabled:
        return rgb
    if params.angle is not None:
        angle = float(params.angle)
        if report is not None:
            report["deskew"] = f"angle impose {angle:+.2f} deg"
        return rotate_image(rgb, angle, crop=params.crop) if abs(angle) > 1e-3 else rgb

    if estimate is None:
        estimate = estimate_tilt(rgb, params)
    if report is not None:
        report["deskew_analysis"] = estimate.describe()
    if abs(estimate.angle) < params.min_angle:
        if report is not None:
            report["deskew"] = f"ignore : deja droite ({estimate.angle:+.2f} deg)"
        return rgb
    if estimate.confidence < params.min_confidence:
        if report is not None:
            report["deskew"] = (f"ignore : confiance insuffisante "
                                f"({estimate.confidence:.2f} < {params.min_confidence})")
        return rgb
    out = rotate_image(rgb, estimate.angle, crop=params.crop)
    if report is not None:
        kept = (out.shape[0] * out.shape[1]) / float(rgb.shape[0] * rgb.shape[1])
        report["deskew"] = (f"redressee de {estimate.angle:+.2f} deg "
                            f"(confiance {estimate.confidence:.2f}, "
                            f"{kept * 100:.0f} % de l'image conservee)")
    return out
