"""Mesure du bruit reellement present dans l'image, niveau par niveau.

Pourquoi ne pas se contenter d'un ecart-type global : le bruit d'un capteur
depend du signal (``var = a*I + b`` : photons + electronique) et le
dematricage le rend correle spatialement, donc son niveau ne decroit pas comme
celui d'un bruit blanc quand on monte d'echelle. Un seuil unique sur-lisse les
ombres ou laisse du bruit dans les hautes lumieres.

naturaw mesure donc, pour chaque canal et chaque niveau de la pyramide, une
droite ``var_j(I) = a_j * I + b_j`` ou ``I`` est la luminosite locale. La
mesure se fait sur le percentile bas des blocs (les plus plats de l'image), ce
qui separe le bruit de la texture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .pyramid import analyse, gaussian_stack, match_size

_MAD_TO_SIGMA = 1.4826
_MAX_BLOCKS = 4000

# Decroissance du bruit d'un niveau de pyramide au suivant, en variance.
# Mesuree sur des captures simulees (scene -> mosaique Bayer -> bruit de
# photons -> dematricage) : le premier passage enleve l'essentiel du bruit
# haute frequence (x0,08), les suivants se comportent comme du bruit blanc
# (x0,25). Le resultat est remarquablement stable d'une scene et d'une
# sensibilite a l'autre (0,05 a 0,10 puis 0,20 a 0,27 sur toute la plage).
#
# Pourquoi predire plutot que mesurer : aux niveaux grossiers d'une image
# riche en texture, *aucun* bloc n'est vide de structure, et la mesure
# surestime le bruit jusqu'a vingt fois -- ce qui efface le detail. La mesure
# reste utilisee comme garde-fou quand elle est *plus basse* que la
# prediction (fichier deja debruite par le boitier, par exemple).
_DECAY_FIRST = 0.08
_DECAY_NEXT = 0.25
_MEASURE_TOLERANCE = 1.5


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
    """Profil de bruit complet d'une image (un modele par canal et par niveau)."""

    channels: List[List[ScaleNoise]] = field(default_factory=list)
    labels: Sequence[str] = ("Y", "Co", "Cg")

    def sigma_report(self, brightness: float = 0.18) -> str:
        parts = []
        for name, scales in zip(self.labels, self.channels):
            if scales:
                parts.append(f"{name}={scales[0].sigma_at(brightness) * 100:.2f}%")
        return "  ".join(parts)

    def luma_sigma(self, brightness: float = 0.18) -> float:
        if not self.channels or not self.channels[0]:
            return 0.0
        return self.channels[0][0].sigma_at(brightness)


def _block_stats(detail: np.ndarray, brightness: np.ndarray, block: int):
    """(sigma, luminosite) par bloc, sur une grille eventuellement echantillonnee."""
    h, w = detail.shape[:2]
    nby, nbx = h // block, w // block
    if nby < 2 or nbx < 2:
        return None, None
    # on ne regarde qu'un sous-ensemble de blocs quand l'image est grande :
    # quelques milliers suffisent pour une statistique, et cela evite de
    # mediane-r des dizaines de millions de pixels
    step = max(1, int(np.ceil(np.sqrt(nby * nbx / _MAX_BLOCKS))))
    d = detail[: nby * block, : nbx * block].reshape(nby, block, nbx, block)
    d = d[::step, :, ::step, :]
    b = brightness[: nby * block, : nbx * block].reshape(nby, block, nbx, block)
    b = b[::step, :, ::step, :]
    n = d.shape[0] * d.shape[2]
    d = np.abs(d.transpose(0, 2, 1, 3).reshape(n, block * block))
    sigma = _MAD_TO_SIGMA * np.median(d, axis=1)
    bmean = b.mean(axis=(1, 3)).reshape(-1)
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
    if a < 0.0:                      # pente negative : impossible physiquement
        return 0.0, max(my, 0.0)
    if b < 0.0:                      # ordonnee negative : on refait a b = 0
        denom = float((weights * x * x).sum())
        a = float((weights * x * y).sum() / denom) if denom > 1e-12 else 0.0
        return max(a, 0.0), 0.0
    return a, b


def estimate_level_noise(details: Sequence[np.ndarray],
                         brightness_stack: Sequence[np.ndarray],
                         block: int = 20, bins: int = 16,
                         percentile: float = 8.0, min_blocks: int = 6,
                         measure_levels: int = 4) -> List[ScaleNoise]:
    """Mesure ``var_j(I)`` pour chaque plan de details d'un canal."""
    models: List[ScaleNoise] = []
    n_meas = max(1, min(int(measure_levels), len(details)))
    for j, detail in enumerate(details):
        if j >= n_meas:
            break
        bright = match_size(brightness_stack[min(j, len(brightness_stack) - 1)],
                            detail.shape)
        bmax = max(float(np.percentile(bright, 99.5)), 1e-3)
        sigma, bmean = _block_stats(detail, bright, block)
        if sigma is None:
            s = _MAD_TO_SIGMA * float(np.median(np.abs(detail)))
            models.append(ScaleNoise(0.0, s * s, measured=False))
        else:
            edges = np.linspace(0.0, bmax, bins + 1)
            idx = np.clip(np.digitize(bmean, edges) - 1, 0, bins - 1)
            xs, ys, ws = [], [], []
            for k in range(bins):
                sel = sigma[idx == k]
                if sel.size < min_blocks:
                    continue
                xs.append(0.5 * (edges[k] + edges[k + 1]))
                ys.append(float(np.percentile(sel, percentile) ** 2))
                ws.append(np.sqrt(float(sel.size)))
            if xs:
                a, b = _fit_line(np.asarray(xs, np.float64), np.asarray(ys, np.float64),
                                 np.asarray(ws, np.float64))
                models.append(ScaleNoise(a, b, measured=True))
            else:
                var = float(np.percentile(sigma, percentile) ** 2)
                models.append(ScaleNoise(0.0, var, measured=False))
        if len(models) >= 2:
            # on borne par la prediction issue du niveau precedent
            prev, last = models[-2], models[-1]
            ratio = _DECAY_FIRST if len(models) == 2 else _DECAY_NEXT
            ceiling = ratio * _MEASURE_TOLERANCE
            last.a = min(last.a, prev.a * ceiling)
            last.b = min(last.b, prev.b * ceiling)
            last.measured = last.measured and (last.a > 0 or last.b > 0)
    return extend_models(models, len(details))


def extend_models(models: List[ScaleNoise], levels: int) -> List[ScaleNoise]:
    """Prolonge la liste jusqu'a ``levels`` niveaux (decroissance mesuree)."""
    out = list(models[:levels])
    if not out:
        return [ScaleNoise(0.0, 0.0, measured=False) for _ in range(levels)]
    while len(out) < levels:
        ref = out[-1]
        ratio = _DECAY_FIRST if len(out) == 1 else _DECAY_NEXT
        out.append(ScaleNoise(ref.a * ratio, ref.b * ratio, measured=False))
    return out


def quick_sigma(rgb: np.ndarray, max_pixels: int = 2_000_000) -> float:
    """Estimation rapide et robuste de l'ecart-type du bruit (sous-echantillonnee).

    Sert avant la mesure complete, quand une etape a besoin d'un ordre de
    grandeur (detection des pixels chauds, par exemple).
    """
    import cv2
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., min(1, arr.shape[2] - 1)]
    step = max(1, int(np.sqrt(arr.size / max(max_pixels, 1))))
    small = np.ascontiguousarray(arr[::step, ::step])
    if min(small.shape) < 8:
        small = arr
    # laplacien normalise : insensible aux rampes, sensible au bruit
    hp = cv2.filter2D(small, -1, np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]],
                                          np.float32) / 6.0)
    return float(_MAD_TO_SIGMA * np.median(np.abs(hp)))


def estimate_scale_noise(channel: np.ndarray, brightness: np.ndarray, levels: int,
                         **kwargs) -> List[ScaleNoise]:
    """Mesure le bruit d'un canal isole (construit sa propre pyramide)."""
    details, _res = analyse(channel, levels)
    stack = gaussian_stack(brightness, levels)
    return estimate_level_noise(details, stack, measure_levels=kwargs.pop("measure_levels", levels),
                                **kwargs)


def measure_profile(ycocg: np.ndarray, levels: int, measure_levels: int = 4,
                    block: int = 20,
                    details: Optional[Sequence[Sequence[np.ndarray]]] = None,
                    brightness_stack: Optional[Sequence[np.ndarray]] = None) -> NoiseProfile:
    """Profil de bruit des trois canaux YCoCg d'une image lineaire.

    ``details`` et ``brightness_stack`` permettent de reutiliser une
    decomposition deja calculee (le debruiteur le fait : la pyramide n'est
    construite qu'une fois par canal).
    """
    if brightness_stack is None:
        brightness_stack = gaussian_stack(np.clip(ycocg[..., 0], 0.0, None), levels)
    channels = []
    for c in range(3):
        if details is not None:
            chan_details = details[c]
        else:
            chan_details, _res = analyse(ycocg[..., c], levels)
        channels.append(estimate_level_noise(chan_details, brightness_stack, block=block,
                                             measure_levels=measure_levels))
    return NoiseProfile(channels=channels)
