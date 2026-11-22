"""Briques de filtrage : filtres rapides et transformee multi-echelle starlet.

La transformee "starlet" (algorithme a trous avec le noyau B3-spline) est
preferee aux ondelettes decimees : elle est isotrope, invariante par
translation et sa reconstruction est une simple somme. Resultat : aucun
artefact en damier ni ringing autour des contours, ce qui est indispensable
pour un debruitage qui doit rester invisible.
"""

from __future__ import annotations

import cv2
import numpy as np

# Noyau B3-spline du starlet
_H = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float32) / 16.0

# Ecart-type residuel d'un bruit blanc unitaire a chaque echelle starlet.
# Sert uniquement de garde-fou / extrapolation : le moteur mesure le bruit
# reellement present a chaque echelle (cf. naturaw.noise).
STARLET_SIGMA = (0.8907, 0.2007, 0.0856, 0.0413, 0.0205, 0.0103, 0.0052, 0.0026)

_BORDER = cv2.BORDER_REFLECT


def box(img: np.ndarray, radius: int) -> np.ndarray:
    """Moyenne locale sur une fenetre carree (filtre separable O(1) d'OpenCV)."""
    radius = max(int(radius), 0)
    if radius == 0:
        return img
    k = 2 * radius + 1
    return cv2.boxFilter(img, -1, (k, k), normalize=True, borderType=_BORDER)


def dilated_kernel(level: int) -> np.ndarray:
    """Noyau B3-spline dilate de 2**level (trous intercales de zeros)."""
    step = 1 << int(level)
    if step == 1:
        return _H
    k = np.zeros(((_H.size - 1) * step + 1,), dtype=np.float32)
    k[::step] = _H
    return k


def starlet_smooth(img: np.ndarray, level: int) -> np.ndarray:
    """Lissage B3-spline a l'echelle ``level`` (0 = plus fine)."""
    k = dilated_kernel(level)
    return cv2.sepFilter2D(img, -1, k, k, borderType=_BORDER)


def starlet_decompose(img: np.ndarray, levels: int):
    """Decompose en ``levels`` plans de details + residu lisse.

    Reconstruction exacte : ``sum(details) + residual``.
    """
    details = []
    cur = img.astype(np.float32, copy=True)
    for j in range(levels):
        nxt = starlet_smooth(cur, j)
        details.append(cur - nxt)
        cur = nxt
    return details, cur


def local_range(src: np.ndarray, radius: int):
    """Minimum et maximum locaux sur une fenetre carree."""
    radius = max(int(radius), 1)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1, 2 * radius + 1))
    lo = cv2.erode(src, k, borderType=_BORDER)
    hi = cv2.dilate(src, k, borderType=_BORDER)
    return lo, hi


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float,
                  max_slope: float = 0.0, clamp: bool = False) -> np.ndarray:
    """Filtre guide (He et al.) : lisse ``src`` en suivant les contours de ``guide``.

    Utilise pour nettoyer la chrominance sans bavure de couleur : la structure
    vient de la luminance, qui est beaucoup moins bruitee.

    ``max_slope`` borne le coefficient de regression et ``clamp`` borne la
    sortie a l'intervalle local de ``src`` : sans ces deux garde-fous, le
    modele lineaire *extrapole* sur un contour tres contraste et peut creer une
    dominante de couleur franche la ou il n'y en avait aucune.
    """
    radius = max(int(radius), 1)
    guide = guide.astype(np.float32, copy=False)
    src = src.astype(np.float32, copy=False)
    mean_i = box(guide, radius)
    mean_p = box(src, radius)
    corr_i = box(guide * guide, radius)
    corr_ip = box(guide * src, radius)
    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + float(eps))
    if max_slope and max_slope > 0.0:
        np.clip(a, -float(max_slope), float(max_slope), out=a)
    b = mean_p - a * mean_i
    out = box(a, radius) * guide + box(b, radius)
    if clamp:
        lo, hi = local_range(src, radius)
        np.clip(out, lo, hi, out=out)
    return out


def cross_bilateral(guide: np.ndarray, src: np.ndarray, radius: int,
                    sigma_range: float, sigma_spatial: float = 0.0) -> np.ndarray:
    """Bilateral joint : lisse ``src`` en se servant de ``guide`` pour les poids.

    Contrairement au filtre guide, la sortie est une *combinaison convexe* des
    valeurs voisines : elle reste donc bornee par le voisinage et ne peut pas
    inventer une couleur absente de l'image (pas de depassement sur un contour
    tres contraste). Les poids utilisent une fonction bi-carree a support
    compact (moins couteuse qu'une exponentielle et plus nette sur les bords).
    """
    radius = max(int(radius), 1)
    guide = np.ascontiguousarray(guide, dtype=np.float32)
    src = np.ascontiguousarray(src, dtype=np.float32)
    h, w = guide.shape[:2]
    g_pad = cv2.copyMakeBorder(guide, radius, radius, radius, radius, _BORDER)
    s_pad = cv2.copyMakeBorder(src, radius, radius, radius, radius, _BORDER)
    if sigma_spatial <= 0:
        sigma_spatial = radius / 1.5
    inv_r2 = 1.0 / max(sigma_range * sigma_range, 1e-12)
    acc = np.zeros_like(src)
    wsum = np.zeros_like(src)
    tmp = np.empty_like(src)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ws = float(np.exp(-0.5 * (dy * dy + dx * dx) / (sigma_spatial ** 2)))
            if ws < 0.02:
                continue
            gs = g_pad[radius + dy: radius + dy + h, radius + dx: radius + dx + w]
            ss = s_pad[radius + dy: radius + dy + h, radius + dx: radius + dx + w]
            np.subtract(gs, guide, out=tmp)
            np.multiply(tmp, tmp, out=tmp)
            np.multiply(tmp, inv_r2, out=tmp)
            np.subtract(1.0, tmp, out=tmp)
            np.maximum(tmp, 0.0, out=tmp)
            np.multiply(tmp, tmp, out=tmp)          # poids bi-carre
            if ws != 1.0:
                np.multiply(tmp, ws, out=tmp)
            acc += tmp * ss
            wsum += tmp
    np.maximum(wsum, 1e-8, out=wsum)
    return acc / wsum


def gradient_magnitude(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Norme du gradient apres un leger flou gaussien (robuste au bruit)."""
    if sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), float(sigma), borderType=_BORDER)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def smoothstep(x: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    """Rampe lisse de 0 a 1 entre ``edge0`` et ``edge1``."""
    if edge1 <= edge0:
        return (x > edge0).astype(np.float32)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32, copy=False)


def resize_long_side(img: np.ndarray, long_side: int, interpolation: int = cv2.INTER_AREA):
    """Redimensionne pour que le plus grand cote vaille ``long_side``.

    Renvoie ``(image, facteur)``; ne fait rien si l'image est deja plus petite.
    """
    h, w = img.shape[:2]
    cur = max(h, w)
    if cur <= long_side:
        return img, 1.0
    scale = long_side / float(cur)
    out = cv2.resize(img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                     interpolation=interpolation)
    return out, scale
