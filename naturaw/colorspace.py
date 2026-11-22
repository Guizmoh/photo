"""Conversions colorimetriques utilisees par le moteur.

Tout le traitement se fait en lumiere *lineaire* (pas de gamma), car c'est le
seul espace ou le modele de bruit du capteur (Poisson + gaussien) est valable
et ou la deconvolution est physiquement correcte.

Pour le debruitage on separe luminance et chrominance avec la transformee
YCoCg, qui est exactement reversible, tres peu couteuse et bien decorrelee :
cela permet de lisser fortement la couleur (ou le bruit est le plus visible et
le detail le moins utile) tout en touchant tres peu la luminance, ce qui est la
cle d'un rendu naturel.
"""

from __future__ import annotations

import cv2
import numpy as np

# Primaires sRGB / Rec.709
LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)

# Matrices YCoCg (appliquees avec cv2.transform : SIMD et multi-thread, soit
# environ deux fois plus rapide qu'une expression numpy canal par canal)
_RGB_TO_YCOCG = np.array([[0.25, 0.5, 0.25],
                          [0.5, 0.0, -0.5],
                          [-0.25, 0.5, -0.25]], dtype=np.float32)
_YCOCG_TO_RGB = np.array([[1.0, 1.0, -1.0],
                          [1.0, 0.0, 1.0],
                          [1.0, -1.0, -1.0]], dtype=np.float32)


def rgb_to_ycocg(rgb: np.ndarray) -> np.ndarray:
    """RGB lineaire -> YCoCg (float32, meme forme)."""
    return cv2.transform(np.ascontiguousarray(rgb, dtype=np.float32), _RGB_TO_YCOCG)


def ycocg_to_rgb(ycc: np.ndarray) -> np.ndarray:
    """YCoCg -> RGB lineaire (inverse exacte de :func:`rgb_to_ycocg`)."""
    return cv2.transform(np.ascontiguousarray(ycc, dtype=np.float32), _YCOCG_TO_RGB)


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Luminance Rec.709 d'une image RGB lineaire."""
    m = np.asarray([LUMA_WEIGHTS], dtype=np.float32)
    out = cv2.transform(np.ascontiguousarray(rgb, dtype=np.float32), m)
    # cv2.transform renvoie (h, w) ou (h, w, 1) selon les versions
    return out if out.ndim == 2 else out[..., 0]


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Courbe de transfert sRGB (encodage), entree/sortie en [0, 1]."""
    x = np.clip(x, 0.0, 1.0)
    out = np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(np.maximum(x, 1e-8), 1.0 / 2.4) - 0.055)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """Inverse de :func:`linear_to_srgb`."""
    x = np.clip(x, 0.0, 1.0)
    out = np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))
    return out.astype(np.float32, copy=False)


def apply_gamma(x: np.ndarray, mode: str = "srgb") -> np.ndarray:
    """Encode la sortie : ``srgb``, ``rec709``, ``gamma22`` ou ``linear``."""
    mode = (mode or "srgb").lower()
    if mode in ("linear", "none"):
        return np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)
    if mode == "srgb":
        return linear_to_srgb(x)
    if mode == "rec709":
        x = np.clip(x, 0.0, 1.0)
        out = np.where(x < 0.018, 4.5 * x, 1.099 * np.power(np.maximum(x, 1e-8), 0.45) - 0.099)
        return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
    if mode in ("gamma22", "22"):
        return np.power(np.clip(x, 0.0, 1.0), 1.0 / 2.2).astype(np.float32, copy=False)
    raise ValueError(f"courbe de transfert inconnue : {mode!r}")
