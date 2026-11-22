"""Defauts qu'un debruiteur ne corrige pas, et qui rendent une photo inutilisable.

Deux problemes echappent completement au filtrage multi-echelle :

* les **pixels chauds** et le bruit impulsionnel (poses longues, forts ISO,
  capteur qui chauffe). Ce sont des points isoles tres ecartes de leurs
  voisins : le debruiteur les prend pour du detail et les garde, et la
  deconvolution les transforme en petites etoiles ;
* les **bandes de lecture** du capteur : un decalage commun a toute une ligne
  ou a toute une colonne. Comme il est parfaitement correle sur des milliers
  de pixels, aucun voisinage local ne peut le voir — mais l'oeil, lui, le voit
  tout de suite des qu'on releve les ombres.

Les deux corrections sont mesurees puis bornees : elles ne s'appliquent que si
le defaut depasse nettement ce que le bruit normal explique.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

_NEIGHBOURS = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
_BORDER = cv2.BORDER_REFLECT


@dataclass
class RescueParams:
    """Reglages des corrections de sauvetage."""

    impulse: str = "auto"
    """``auto``, ``on`` ou ``off`` : suppression des pixels chauds/morts."""

    impulse_threshold: float = 6.0
    """Ecart minimal au voisinage, en ecarts-types de bruit."""

    banding: str = "off"
    """``off`` (defaut), ``auto``, ``h``, ``v`` ou ``hv`` : correction des bandes.

    Volontairement desactivee par defaut : une scene dont les structures sont
    parfaitement alignees sur la grille du capteur (facade, mire) ressemble
    beaucoup a des bandes de lecture, et la correction y ferait des degats.
    A activer quand on *voit* des rayures (poses longues, hauts ISO, certains
    boitiers en lecture rapide)."""

    banding_strength: float = 1.0
    """Dose de correction des bandes (1 = correction complete)."""


def _impulse_rate(rgb: np.ndarray, sigma: float, threshold: float) -> float:
    """Taux de pixels aberrants estime sur un echantillon de l'image.

    Une passe complete sur 24 Mpx coute presque une seconde ; la plupart des
    fichiers n'ont aucun pixel chaud. On echantillonne donc d'abord.
    """
    h, w = rgb.shape[:2]
    if h < 64 or w < 64:
        return 1.0
    ys = slice(h // 4, 3 * h // 4)
    xs = slice(w // 4, 3 * w // 4)
    sample = np.ascontiguousarray(rgb[ys, xs, 1])
    if sample.size > 1_000_000:
        step = int(np.sqrt(sample.size / 1_000_000)) + 1
        sample = np.ascontiguousarray(sample[::step, ::step])
    hi = cv2.dilate(sample, _NEIGHBOURS, borderType=_BORDER)
    lo = cv2.erode(sample, _NEIGHBOURS, borderType=_BORDER)
    local = threshold * sigma * (1.0 + 1.1 * np.maximum(sample, 0.0))
    bad = (sample > hi + local) | (sample < lo - local)
    return float(np.count_nonzero(bad)) / max(sample.size, 1)


def remove_impulse(rgb: np.ndarray, sigma: float, threshold: float = 6.0,
                   report: Optional[dict] = None, check_first: bool = True) -> np.ndarray:
    """Remplace les pixels isoles aberrants par la mediane de leur voisinage.

    Un pixel n'est corrige que s'il sort de *tout* son voisinage immediat : on
    le compare au maximum et au minimum des huit voisins (centre exclu), pas a
    une moyenne. Un detail fin, meme tres contraste, est porte par plusieurs
    pixels et survit ; un pixel chaud, non.
    """
    rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    sigma = max(float(sigma), 1e-4)
    limit = threshold * sigma
    if check_first and _impulse_rate(rgb, sigma, threshold) < 2e-6:
        if report is not None:
            report["pixels_chauds"] = "aucun pixel aberrant"
        return rgb
    out = rgb.copy()
    total = 0
    for c in range(rgb.shape[2]):
        chan = rgb[..., c]
        hi = cv2.dilate(chan, _NEIGHBOURS, borderType=_BORDER)     # max des voisins
        lo = cv2.erode(chan, _NEIGHBOURS, borderType=_BORDER)      # min des voisins
        # Le seuil suit le bruit local (plus clair = plus bruite). On evite
        # une racine carree pleine resolution avec une borne lineaire, qui
        # majore sqrt(1 + 4I) sur [0, 1] : un peu plus prudente, gratuite.
        local = (limit * (1.0 + 1.1 * np.maximum(chan, 0.0))).astype(np.float32)
        np.add(hi, local, out=hi)
        np.subtract(lo, local, out=lo)
        bad = (chan > hi) | (chan < lo)
        n = int(np.count_nonzero(bad))
        if n:
            med = cv2.medianBlur(chan, 3)
            out[..., c] = np.where(bad, med, chan)
        total += n
    if report is not None:
        ratio = total / float(rgb.shape[0] * rgb.shape[1] * rgb.shape[2])
        report["pixels_chauds"] = f"{total} pixels corriges ({ratio * 100:.4f} %)"
    return out


def _detrend(x: np.ndarray, window: int = 31) -> np.ndarray:
    """Retire la variation lente d'une suite de decalages.

    Une bande de lecture change d'une ligne a l'autre ; le contenu de l'image,
    lui, varie doucement. On ne garde donc que la partie "nerveuse".
    """
    k = max(3, int(window) | 1)
    smooth = cv2.blur(x.reshape(-1, 1), (1, k), borderType=_BORDER).ravel()
    return (x - smooth).astype(np.float32)


def _flat_mask(chan: np.ndarray, keep: float = 0.5) -> np.ndarray:
    """Pixels les plus plats de l'image : les seuls ou un decalage se lit.

    Sur un contour, l'ecart a la reference lissee vient du contour lui-meme,
    pas du capteur. On ne mesure donc que sur la moitie la plus lisse.
    """
    g = cv2.magnitude(cv2.Sobel(chan, cv2.CV_32F, 1, 0, ksize=3),
                      cv2.Sobel(chan, cv2.CV_32F, 0, 1, ksize=3))
    g = cv2.blur(g, (3, 3), borderType=_BORDER)
    return g <= float(np.quantile(g[::4, ::4], keep))


def _offsets_halves(chan: np.ndarray, axis: int):
    """Decalage median de chaque ligne, estime separement sur ses deux moities.

    ``axis=0`` : lignes (decalage horizontal commun) ; ``axis=1`` : colonnes.
    Le lissage de reference se fait *dans l'autre direction*, pour que les
    lignes voisines servent de reference a la ligne courante.
    """
    if axis == 0:   # bandes horizontales : lisser verticalement
        hp = chan - cv2.blur(chan, (1, 65), borderType=_BORDER)
    else:           # bandes verticales : lisser horizontalement
        hp = chan - cv2.blur(chan, (65, 1), borderType=_BORDER)
    masked = np.where(_flat_mask(chan), hp, np.nan)
    with np.errstate(invalid="ignore"):
        if axis == 0:
            half = masked.shape[1] // 2
            a = np.nanmedian(masked[:, :half], axis=1)
            b = np.nanmedian(masked[:, half:], axis=1)
        else:
            half = masked.shape[0] // 2
            a = np.nanmedian(masked[:half, :], axis=0)
            b = np.nanmedian(masked[half:, :], axis=0)
    a = np.nan_to_num(a).astype(np.float32)
    b = np.nan_to_num(b).astype(np.float32)
    return _detrend(a), _detrend(b)


def _banding_estimate(chan: np.ndarray, axis: int):
    """(correlation des deux moities, decalage estime) pour un canal.

    Si une ligne porte vraiment un decalage de lecture, ses deux moities le
    voient toutes les deux : leurs estimations sont correlees. Si ce n'est que
    du bruit ou de la texture, elles sont independantes. La correlation ``r``
    sert alors de retrecissement optimal (Wiener) : ``2r / (1 + r)`` applique
    a la moyenne des deux moities. Aucune detection a seuil arbitraire, et sur
    une image sans bandes la correction tend d'elle-meme vers zero.
    """
    a, b = _offsets_halves(chan, axis)
    if a.size < 8:
        return 0.0, np.zeros_like(a)
    sa, sb = float(np.std(a)), float(np.std(b))
    if sa < 1e-9 or sb < 1e-9:
        return 0.0, np.zeros_like(a)
    r = float(np.clip(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb), 0.0, 1.0))
    mean = 0.5 * (a + b)
    # les rares lignes ou un objet occupe plus de la moitie de la largeur
    # donnent un decalage enorme : on les ecrete a quelques ecarts robustes
    mad = 1.4826 * float(np.median(np.abs(mean - np.median(mean)))) + 1e-9
    np.clip(mean, -3.0 * mad, 3.0 * mad, out=mean)
    shrink = 2.0 * r / (1.0 + r)
    return r, (shrink * mean).astype(np.float32)


def measure_banding(luma: np.ndarray, sigma: float = 0.0):
    """Correlation des deux moities pour les lignes et pour les colonnes.

    Proche de 0 : pas de bandes. Au-dela de ~0,45 : structure de lecture
    probable. ``sigma`` n'est pas utilise (garde pour compatibilite).
    """
    luma = np.ascontiguousarray(luma, dtype=np.float32)
    return _banding_estimate(luma, 0)[0], _banding_estimate(luma, 1)[0]


def remove_banding(rgb: np.ndarray, sigma: float = 0.0, mode: str = "off",
                   strength: float = 1.0, detect_ratio: float = 0.45,
                   report: Optional[dict] = None) -> np.ndarray:
    """Retire les bandes horizontales et/ou verticales du capteur.

    L'estimation est faite canal par canal (les amplificateurs de lecture
    different d'un canal a l'autre) et retrecie par la correlation mesuree :
    sur une image sans bandes, la correction vaut pratiquement zero.
    """
    mode = (mode or "auto").lower()
    if mode in ("off", "non", "none") or strength <= 0.0:
        return rgb
    rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    want = {"auto": (True, True), "h": (True, False), "v": (False, True),
            "hv": (True, True)}.get(mode, (True, True))
    out = rgb.copy()
    scores = {0: [], 1: []}
    for c in range(out.shape[2]):
        chan = out[..., c]
        for axis, enabled in ((0, want[0]), (1, want[1])):
            if not enabled:
                continue
            r, offsets = _banding_estimate(chan, axis)
            scores[axis].append(r)
            if mode == "auto" and r < detect_ratio:
                continue
            if axis == 0:
                chan -= strength * offsets[:, None]
            else:
                chan -= strength * offsets[None, :]
    np.clip(out, 0.0, None, out=out)
    if report is not None:
        rows = max(scores[0]) if scores[0] else 0.0
        cols = max(scores[1]) if scores[1] else 0.0
        report["bandes_mesure"] = f"lignes r={rows:.2f}, colonnes r={cols:.2f}"
        acted = []
        if rows >= detect_ratio or mode in ("h", "hv"):
            acted.append("lignes")
        if cols >= detect_ratio or mode in ("v", "hv"):
            acted.append("colonnes")
        report["bandes"] = ("bandes corrigees : " + " et ".join(acted)) if acted \
            else "aucune bande detectee"
    return out


def apply_rescue(rgb: np.ndarray, params: RescueParams, sigma: float,
                 report: Optional[dict] = None) -> np.ndarray:
    """Enchaine les corrections de sauvetage disponibles."""
    mode = (params.impulse or "auto").lower()
    if mode not in ("off", "non", "none"):
        rgb = remove_impulse(rgb, sigma, params.impulse_threshold, report=report)
    rgb = remove_banding(rgb, sigma, params.banding, params.banding_strength,
                         report=report)
    return rgb
