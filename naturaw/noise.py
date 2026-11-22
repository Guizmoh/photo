"""Mesure du bruit reellement present dans l'image, echelle par echelle.

Pourquoi ne pas se contenter d'un ecart-type global : le bruit d'un capteur est
*dependant du signal* (var = a*I + b : photons + electronique), et le
dematricage le rend *correle spatialement*, donc son niveau ne decroit pas
comme celui d'un bruit blanc quand on monte d'echelle. Un seuil unique
sur-lisse les ombres ou laisse du bruit dans les hautes lumieres.

Le moteur mesure donc, pour chaque plan de details starlet et chaque canal, une
droite ``var_j(I) = a_j * I + b_j`` ou ``I`` est la luminosite locale. La mesure
se fait sur le percentile bas des blocs de l'image (les blocs les plus plats),
ce qui separe le bruit de la texture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

from .filters import STARLET_SIGMA, box, starlet_smooth

_MAD_TO_SIGMA = 1.4826


@dataclass
class ScaleNoise:
    """Modele ``var = a * I + b`` pour un plan de details."""

    a: float
    b: float
    measured: bool = True

    def variance(self, brightness: np.ndarray) -> np.ndarray:
        v = self.a * np.clip(brightness, 0.0, None) + self.b
        return np.maximum(v, 0.0, out=v if isinstance(v, np.ndarray) else None)

    def sigma_at(self, brightness: float) -> float:
        return float(np.sqrt(max(self.a * max(brightness, 0.0) + self.b, 0.0)))


@dataclass
class NoiseProfile:
    """Profil de bruit complet d'une image (un modele par canal et par echelle)."""

    channels: List[List[ScaleNoise]] = field(default_factory=list)
    labels: Sequence[str] = ("Y", "Co", "Cg")

    def sigma_report(self, brightness: float = 0.18) -> str:
        parts = []
        for name, scales in zip(self.labels, self.channels):
            if not scales:
                continue
            s = scales[0].sigma_at(brightness) * 100.0
            parts.append(f"{name}={s:.2f}%")
        return "  ".join(parts)

    def luma_sigma(self, brightness: float = 0.18) -> float:
        if not self.channels or not self.channels[0]:
            return 0.0
        return self.channels[0][0].sigma_at(brightness)


def _block_stats(detail: np.ndarray, brightness: np.ndarray, block: int):
    """Renvoie (sigma_bloc, luminosite_bloc) pour une grille de blocs."""
    h, w = detail.shape[:2]
    nby, nbx = h // block, w // block
    if nby < 2 or nbx < 2:
        return None, None
    d = detail[: nby * block, : nbx * block].reshape(nby, block, nbx, block)
    d = np.abs(d).transpose(0, 2, 1, 3).reshape(nby * nbx, block * block)
    sigma = _MAD_TO_SIGMA * np.median(d, axis=1)
    bmap = brightness[: nby * block, : nbx * block].reshape(nby, block, nbx, block)
    bmean = bmap.mean(axis=(1, 3)).reshape(-1)
    return sigma.astype(np.float32), bmean.astype(np.float32)


def _fit_line(x: np.ndarray, y: np.ndarray, weights: np.ndarray):
    """Moindres carres ponderes de ``y = a*x + b`` avec a >= 0 et b >= 0."""
    if x.size == 0:
        return 0.0, 0.0
    if x.size == 1:
        return 0.0, float(max(y[0], 0.0))
    wsum = float(weights.sum())
    if wsum <= 0:
        return 0.0, float(max(np.median(y), 0.0))
    mx = float((weights * x).sum() / wsum)
    my = float((weights * y).sum() / wsum)
    sxx = float((weights * (x - mx) ** 2).sum())
    sxy = float((weights * (x - mx) * (y - my)).sum())
    a = sxy / sxx if sxx > 1e-12 else 0.0
    b = my - a * mx
    if a < 0.0:  # pente negative impossible physiquement -> modele constant
        return 0.0, max(my, 0.0)
    if b < 0.0:  # ordonnee negative -> on force b = 0 et on refait la pente
        denom = float((weights * x * x).sum())
        a = float((weights * x * y).sum() / denom) if denom > 1e-12 else 0.0
        return max(a, 0.0), 0.0
    return a, b


def estimate_scale_noise(
    channel: np.ndarray,
    brightness: np.ndarray,
    levels: int,
    block: int = 24,
    bins: int = 16,
    percentile: float = 12.0,
    min_blocks: int = 6,
    decay_cap: float = 0.6,
) -> List[ScaleNoise]:
    """Mesure ``var_j(I)`` pour chaque plan de details d'un canal.

    ``brightness`` est une version lissee de la luminance (variable explicative
    commune a tous les canaux : le bruit depend du nombre de photons, pas de la
    valeur du canal chroma lui-meme).

    Deux garde-fous evitent de confondre texture et bruit :
      * la taille des blocs grandit avec l'echelle (les coefficients y sont
        correles sur de plus grandes distances) ;
      * le bruit mesure doit decroitre d'une echelle a la suivante
        (``decay_cap``). Une texture fine presente partout ferait sinon gonfler
        l'estimation aux echelles grossieres, et donc lisser le detail utile.
    """
    models: List[ScaleNoise] = []
    cur = channel.astype(np.float32, copy=False)
    bmax = float(np.percentile(brightness, 99.5)) if brightness.size else 1.0
    bmax = max(bmax, 1e-3)
    for j in range(levels):
        nxt = starlet_smooth(cur, j)
        detail = cur - nxt
        block_j = int(min(96, block * (1 << j)))
        sigma, bmean = _block_stats(detail, brightness, block_j)
        if sigma is None and block_j > block:
            sigma, bmean = _block_stats(detail, brightness, block)
        if sigma is None:
            # image trop petite : repli sur un MAD global
            s = _MAD_TO_SIGMA * float(np.median(np.abs(detail)))
            models.append(ScaleNoise(0.0, s * s, measured=False))
            cur = nxt
            continue
        edges = np.linspace(0.0, bmax, bins + 1)
        idx = np.clip(np.digitize(bmean, edges) - 1, 0, bins - 1)
        xs, ys, ws = [], [], []
        for k in range(bins):
            sel = sigma[idx == k]
            if sel.size < min_blocks:
                continue
            var = float(np.percentile(sel, percentile) ** 2)
            xs.append(0.5 * (edges[k] + edges[k + 1]))
            ys.append(var)
            ws.append(np.sqrt(float(sel.size)))
        if xs:
            a, b = _fit_line(np.asarray(xs, np.float64), np.asarray(ys, np.float64),
                             np.asarray(ws, np.float64))
            models.append(ScaleNoise(a, b, measured=True))
        else:
            var = float(np.percentile(sigma, percentile) ** 2)
            models.append(ScaleNoise(0.0, var, measured=False))
        if len(models) >= 2 and decay_cap > 0:
            prev, last = models[-2], models[-1]
            last.a = min(last.a, prev.a * decay_cap)
            last.b = min(last.b, prev.b * decay_cap)
        cur = nxt
    return models


def extend_models(models: List[ScaleNoise], levels: int) -> List[ScaleNoise]:
    """Complete / prolonge la liste de modeles jusqu'a ``levels`` echelles.

    Au-dela des echelles mesurees on extrapole avec la decroissance theorique
    du starlet, en se calant sur la derniere echelle mesuree.
    """
    out = list(models[:levels])
    if not out:
        return [ScaleNoise(0.0, 0.0, measured=False) for _ in range(levels)]
    while len(out) < levels:
        j = len(out)
        ref = out[-1]
        ratio = (STARLET_SIGMA[min(j, len(STARLET_SIGMA) - 1)] /
                 STARLET_SIGMA[min(j - 1, len(STARLET_SIGMA) - 1)]) ** 2
        out.append(ScaleNoise(ref.a * ratio, ref.b * ratio, measured=False))
    return out


def measure_profile(ycocg: np.ndarray, levels: int, measure_levels: int = 4,
                    block: int = 24) -> NoiseProfile:
    """Profil de bruit des trois canaux YCoCg d'une image lineaire."""
    brightness = box(ycocg[..., 0], 3)
    np.clip(brightness, 0.0, None, out=brightness)
    n_meas = max(1, min(int(measure_levels), int(levels)))
    channels = []
    for c in range(3):
        models = estimate_scale_noise(ycocg[..., c], brightness, n_meas, block=block)
        channels.append(extend_models(models, levels))
    return NoiseProfile(channels=channels)
