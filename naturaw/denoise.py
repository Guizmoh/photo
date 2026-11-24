"""Debruitage multi-echelle "naturel".

Principe : plutot que de seuiller des coefficients (ce qui laisse des traces
en plaques et un rendu plastique), on applique a chaque niveau de la pyramide
un *filtre de Wiener local* : le gain conserve exactement la part de signal
qui depasse le bruit mesure dans le voisinage.

    gain = max(0, (E - var_bruit) / E)        avec E = energie locale moyenne

Puis une **seconde passe** (Wiener empirique) refait le calcul en prenant
l'estimation de la premiere passe comme reference du signal :

    gain = E_estime / (E_estime + var_bruit)

C'est la meme idee qu'au second etage de BM3D : la premiere passe sait ou se
trouve le signal, la seconde peut donc doser beaucoup plus finement — plus de
bruit enleve *et* plus de texture gardee, pour un cout faible.

Trois choix font le rendu naturel :

1. **Luminance et chrominance separees.** Le bruit chroma (plaques de couleur)
   est beaucoup plus visible et porte beaucoup moins d'information : il est
   traite une fois et demie plus fort. La luminance, elle, est a peine touchee
   -> la nettete subjective est conservee.
2. **Protection des zones texturees.** Un masque de structure, calcule a
   chaque niveau, reinjecte une part des coefficients la ou le contraste local
   domine clairement le bruit : feuillages, tissus, cheveux, gravier.
3. **Grain conserve.** Une fraction du bruit fin retire est remise dans la
   luminance seulement. Une image totalement lisse se *voit* ; un grain fin
   residuel, achromatique, est ce qui rend le resultat credible.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Sequence

import cv2
import numpy as np

from .colorspace import linear_to_srgb, rgb_to_ycocg, ycocg_to_rgb
from .filters import box, cross_bilateral
from .noise import NoiseProfile, ScaleNoise, measure_profile
from .pyramid import analyse, gaussian_stack, match_size, pyramid_levels, synthese

# Ponderation du seuil par niveau : un peu plus ferme sur les grandes
# echelles, ou le bruit basse frequence se voit en plaques.
_LEVEL_WEIGHT = (1.0, 1.0, 1.06, 1.12, 1.18, 1.24, 1.3, 1.3, 1.3)
_WINDOW = 2          # rayon d'analyse, en pixels du niveau courant
_EPS = 1e-12
# Au-dela de cette taille, les cartes de gain et de structure sont calculees
# en demi-resolution puis interpolees. Elles sont lisses par construction
# (moyennes locales), donc la difference est invisible, et cela divise par
# quatre le cout du niveau le plus fin, qui porte 75 % des pixels.
_HALFRES_ABOVE = 2_000_000


@dataclass
class DenoiseParams:
    """Reglages du debruitage (tous relatifs : 1.0 = reglage nominal)."""

    strength: float = 1.0
    """Force globale. 0 desactive, 0.6 = discret, 1.6 = fort."""

    chroma: float = 1.7
    """Multiplicateur de force sur la chrominance.

    Cale sur des captures simulees : a 1,7 l'erreur de chrominance residuelle
    (les plaques de couleur, le defaut le plus visible a haut ISO) baisse de
    10 % par rapport a 1,5, sans lisser les couleurs des textures fines. Du
    bruit blanc ajoute apres dematricage aurait suggere l'inverse."""

    grain: float = 0.18
    """Fraction du grain fin de luminance conservee (0 = lisse, 0.35 = argentique).

    La valeur est automatiquement reduite quand le bruit mesure est fort :
    garder 18 % d'un grain discret rend l'image credible, garder 18 % du bruit
    d'un 25600 ISO ne fait que remettre du bruit. Voir ``grain_floor``."""

    grain_floor: float = 0.3
    """Part minimale du grain conservee quand le bruit est tres fort."""

    detail: float = 0.7
    """Protection des zones texturees (0 = aucune, 1 = maximale).

    Mesure : +1,5 point de MTF de texture par rapport a 0,45, sans perte
    ailleurs."""

    levels: int = 0
    """Nombre de niveaux ; 0 = automatique selon la taille de l'image."""

    refine: bool = True
    """Seconde passe de Wiener empirique (nettement meilleure, cout faible)."""

    auto_strength: bool = True
    """Attenuer la force quand le bruit mesure est negligeable.

    A 100 ISO le bruit ne se voit pas : le debruiter ne gagne rien et coute
    du detail (mesure sur mire a statistiques naturelles : -1 dB). La force
    est donc ramenee progressivement a zero en dessous de 0,8 % d'ecart-type
    a 18 % de gris, et pleine au-dela."""

    chroma_cleanup: float = 0.0
    """Passe supplementaire contre les plaques de couleur (0 = desactivee).

    Le Wiener multi-echelle traite deja le bruit chroma basse frequence, parce
    que le bruit est mesure niveau par niveau. Cette passe n'est utile que sur
    des fichiers ou des taches de couleur subsistent malgre tout ; activee par
    defaut, elle biaisait legerement les degrades propres (un ciel)."""

    safety: float = 5.0
    """Ecart maximal autorise par canal, en ecarts-types de bruit (0 = aucun).

    La borne est evaluee sur une carte de luminosite lissee, calculee en
    demi-resolution : elle est donc respectee a quelques pour cent pres."""

    def scaled(self, factor: float) -> "DenoiseParams":
        out = DenoiseParams(**self.__dict__)
        out.strength = self.strength * factor
        return out


PRESETS = {
    "off": 0.0,
    "leger": 0.6,
    "light": 0.6,
    "normal": 1.0,
    "fort": 1.5,
    "strong": 1.5,
    "extreme": 2.1,
    "sauvetage": 2.6,
    "rescue": 2.6,
}


def auto_levels(shape: Sequence[int], requested: int = 0) -> int:
    """Nombre de niveaux de pyramide adapte a la taille de l'image."""
    return pyramid_levels(shape, requested)


def _workers() -> int:
    """Nombre de fils pour le traitement des canaux.

    Les trois canaux sont independants : les traiter de front fait gagner un
    facteur deux (mesure), car OpenCV et numpy relachent le GIL. On s'aligne
    sur le budget de threads d'OpenCV pour ne pas se marcher dessus quand
    plusieurs photos sont traitees en parallele.
    """
    try:
        return 3 if cv2.getNumThreads() >= 2 else 1
    except Exception:
        return 1


def _map_channels(func, count: int = 3):
    """Applique ``func`` aux canaux, en parallele si le budget le permet."""
    if _workers() <= 1 or count <= 1:
        return [func(c) for c in range(count)]
    with ThreadPoolExecutor(max_workers=count) as pool:
        return list(pool.map(func, range(count)))


def noise_scaled_strength(sigma: float, lo: float = 0.0025, hi: float = 0.008) -> float:
    """Facteur [0,1] applique a la force selon le bruit reellement mesure."""
    if sigma <= lo:
        return 0.0
    t = float(np.clip((sigma - lo) / max(hi - lo, 1e-9), 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def effective_grain(grain: float, sigma: float, floor: float = 0.3) -> float:
    """Grain reellement reinjecte, attenue quand le bruit mesure est fort."""
    if grain <= 0.0:
        return 0.0
    factor = float(np.clip(0.02 / max(sigma, 1e-4), floor, 1.0))
    return grain * factor


def _smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    t *= t * (3.0 - 2.0 * t)
    return t


def _half(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, ((img.shape[1] + 1) // 2, (img.shape[0] + 1) // 2),
                      interpolation=cv2.INTER_AREA)


def _full(img: np.ndarray, shape) -> np.ndarray:
    return cv2.resize(img, (int(shape[1]), int(shape[0])), interpolation=cv2.INTER_LINEAR)


def _local_energy(detail: np.ndarray, reduced: bool):
    """Energie locale moyenne, eventuellement calculee en demi-resolution."""
    sq = detail * detail
    if reduced:
        return box(_half(sq), 1)
    return box(sq, _WINDOW)


def _structure_mask(detail: np.ndarray, variance: np.ndarray,
                    lo: float = 1.8, hi: float = 6.0) -> np.ndarray:
    """Masque [0,1] la ou le contraste local domine clairement le bruit."""
    reduced = detail.size > _HALFRES_ABOVE
    energy = _local_energy(detail, reduced)
    var = _half(variance) if reduced else variance
    ratio = np.sqrt(energy / np.maximum(var, 1e-12))
    mask = _smoothstep(ratio, lo, hi)
    return _full(mask, detail.shape) if reduced else mask


def _wiener_level(detail: np.ndarray, variance: np.ndarray, mask: Optional[np.ndarray],
                  detail_amount: float, refine: bool) -> np.ndarray:
    """Gain de Wiener local (+ seconde passe empirique) sur un plan de details.

    Sur les grands niveaux, les cartes de gain sont calculees en
    demi-resolution : ce sont des moyennes locales deja lissees, et c'est le
    poste de calcul dominant.
    """
    reduced = detail.size > _HALFRES_ABOVE
    energy = _local_energy(detail, reduced)
    var = _half(variance) if reduced else variance
    gain = (energy - var) / np.maximum(energy, _EPS)
    np.clip(gain, 0.0, 1.0, out=gain)
    # la carte de gain est elle-meme bruitee : on la lisse pour eviter
    # d'alterner des plaques nettes et des plaques bruitees
    gain = box(gain, 1 if reduced else _WINDOW)
    msk = (_half(mask) if reduced else mask) if mask is not None else None
    if msk is not None and detail_amount > 0.0:
        # on remonte le gain vers 1 la ou il y a vraiment de la structure.
        # Forme additive et non max(gain, masque) : le masque reste toujours
        # sous le gain de Wiener, un maximum ne ferait donc jamais rien.
        gain += (detail_amount * msk) * (1.0 - gain)
    if refine:
        # Wiener empirique : l'energie du signal est estimee sur la premiere
        # passe, et non plus sur l'image bruitee
        kept_energy = energy * (gain * gain)
        gain = kept_energy / (kept_energy + var + _EPS)
        if msk is not None and detail_amount > 0.0:
            gain += (detail_amount * msk) * (1.0 - gain)
    if reduced:
        gain = _full(gain, detail.shape)
    return detail * gain


def _denoise_channel(details: List[np.ndarray], residual: np.ndarray,
                     models: List[ScaleNoise], bright_stack: Sequence[np.ndarray],
                     strength: float, grain: float = 0.0,
                     masks: Optional[Sequence[np.ndarray]] = None,
                     detail_amount: float = 0.0, refine: bool = True) -> np.ndarray:
    """Applique le Wiener local a tous les niveaux, puis reconstruit."""
    out_levels = []
    for j, detail in enumerate(details):
        bright = match_size(bright_stack[min(j, len(bright_stack) - 1)], detail.shape)
        k = strength * _LEVEL_WEIGHT[min(j, len(_LEVEL_WEIGHT) - 1)]
        variance = models[j].variance(bright) * (k * k)
        mask = masks[j] if masks is not None and j < len(masks) else None
        kept = _wiener_level(detail, variance, mask, detail_amount, refine)
        if grain > 0.0 and j <= 1:
            g = grain if j == 0 else grain * 0.45
            kept += g * (detail - kept)
        out_levels.append(kept)
    return synthese(out_levels, residual)


def _chroma_cleanup(ycc: np.ndarray, profile: NoiseProfile, amount: float) -> None:
    """Supprime les plaques de couleur basse frequence restantes (sur place)."""
    if amount <= 0.0:
        return
    h, w = ycc.shape[:2]
    if min(h, w) < 120:
        return
    factor = 4 if min(h, w) >= 480 else 2
    small = (max(1, w // factor), max(1, h // factor))
    # Le guide est encode en gamma : l'ecart de luminance qui compte
    # perceptuellement est relatif, et le bruit y devient a peu pres uniforme
    # sur toute la plage tonale.
    guide_lo = cv2.resize(linear_to_srgb(np.clip(ycc[..., 0], 0.0, 1.0)), small,
                          interpolation=cv2.INTER_AREA)
    sigma_y = max(profile.channels[0][0].sigma_at(0.18), 2e-3)
    sigma_range = float(np.clip(6.0 * sigma_y, 0.04, 0.25))
    brightness = cv2.resize(np.clip(ycc[..., 0], 0.0, None), small, interpolation=cv2.INTER_AREA)
    brightness = cv2.resize(brightness, (w, h), interpolation=cv2.INTER_LINEAR)
    for c in (1, 2):
        chroma = ycc[..., c]
        lo = cv2.resize(chroma, small, interpolation=cv2.INTER_AREA)
        sm = cross_bilateral(guide_lo, lo, radius=3, sigma_range=sigma_range)
        base = cv2.resize(lo, (w, h), interpolation=cv2.INTER_LINEAR)
        back = cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR)
        delta = (back - base) * amount
        coarse = profile.channels[c][min(2, len(profile.channels[c]) - 1)]
        limit = 2.5 * np.sqrt(coarse.variance(brightness))
        np.clip(delta, -limit, limit, out=delta)
        chroma += delta


def denoise_linear_rgb(
    rgb: np.ndarray,
    params: DenoiseParams = DenoiseParams(),
    profile: Optional[NoiseProfile] = None,
    report: Optional[dict] = None,
) -> np.ndarray:
    """Debruite une image RGB lineaire float32 dans [0, ~1]."""
    if params.strength <= 0.0:
        return rgb
    rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    levels = pyramid_levels(rgb.shape, params.levels)
    ycc = rgb_to_ycocg(rgb)

    bright_stack = gaussian_stack(np.clip(ycc[..., 0], 0.0, None), levels)
    pyramids = _map_channels(lambda c: analyse(ycc[..., c], levels))

    if profile is None:
        profile = measure_profile(ycc, levels, details=[p[0] for p in pyramids],
                                  brightness_stack=bright_stack)
    if report is not None:
        report["noise_levels"] = levels
        report["noise_sigma"] = profile.sigma_report()
        report["noise_luma_sigma"] = profile.luma_sigma()
        report["debruitage"] = (f"force {params.strength:.2f}, chroma x{params.chroma:.1f}, "
                                f"grain {params.grain:.2f}, "
                                f"{'2 passes' if params.refine else '1 passe'}, "
                                f"{levels} niveaux")

    # masques de structure, niveau par niveau, calcules sur la luminance
    luma_masks: List[np.ndarray] = []
    if params.detail > 0.0:
        for j, detail in enumerate(pyramids[0][0]):
            bright = match_size(bright_stack[min(j, len(bright_stack) - 1)], detail.shape)
            luma_masks.append(_structure_mask(detail, profile.channels[0][j].variance(bright)))

    sigma_luma = profile.luma_sigma(0.18)
    strength = params.strength
    if params.auto_strength:
        factor = noise_scaled_strength(sigma_luma)
        strength *= factor
        if report is not None and factor < 0.99:
            report["force_appliquee"] = (f"{strength:.2f} (bruit mesure {sigma_luma * 100:.2f} %"
                                         f" : attenuation x{factor:.2f})")
    if strength <= 1e-3:
        if report is not None:
            report["debruitage"] = (f"inutile : bruit mesure a {sigma_luma * 100:.2f} %, "
                                    "image laissee intacte")
        return rgb
    grain = effective_grain(params.grain, sigma_luma, params.grain_floor)
    if report is not None:
        report["grain_applique"] = round(grain, 3)

    chroma_strength = strength * params.chroma

    def run_channel(c: int) -> np.ndarray:
        details, residual = pyramids[c]
        if c == 0:
            return _denoise_channel(
                details, residual, profile.channels[0], bright_stack,
                strength=strength, grain=grain,
                masks=luma_masks or None, detail_amount=params.detail,
                refine=params.refine)
        masks: Optional[List[np.ndarray]] = None
        if params.detail > 0.0:
            masks = []
            for j, detail in enumerate(details):
                bright = match_size(bright_stack[min(j, len(bright_stack) - 1)], detail.shape)
                cmask = _structure_mask(detail, profile.channels[c][j].variance(bright),
                                        lo=3.0, hi=8.0)
                if j < len(luma_masks):
                    np.maximum(cmask, 0.3 * luma_masks[j], out=cmask)
                masks.append(cmask)
        return _denoise_channel(
            details, residual, profile.channels[c], bright_stack,
            strength=chroma_strength, grain=0.0, masks=masks,
            detail_amount=0.9 * max(params.detail, 0.2), refine=params.refine)

    out = np.empty_like(ycc)
    for c, chan in enumerate(_map_channels(run_channel)):
        out[..., c] = chan
    # on libere les pyramides (trois fois la taille de l'image) sans delier
    # les noms, auxquels la fermeture ci-dessus fait reference
    pyramids.clear()
    luma_masks.clear()

    _chroma_cleanup(out, profile, float(np.clip(params.chroma_cleanup, 0.0, 1.0)))

    if params.safety > 0.0:
        # Filet de securite, applique en dernier : aucun canal ne s'ecarte de
        # l'original de plus de ``safety`` ecarts-types de bruit. Un debruiteur
        # multi-echelle peut sinon, au voisinage d'un aplat tres sature,
        # reconstituer une couleur franche a partir d'un niveau grossier.
        # la borne suit la luminosite, qui est lisse : on la calcule en
        # demi-resolution (racine carree comprise) puis on l'interpole
        bright_small = bright_stack[1] if len(bright_stack) > 1 else bright_stack[0]
        for c in range(3):
            limit = params.safety * np.sqrt(profile.channels[c][0].variance(bright_small))
            limit = _full(limit, out.shape[:2])
            np.clip(out[..., c], ycc[..., c] - limit, ycc[..., c] + limit,
                    out=out[..., c])

    rgb_out = ycocg_to_rgb(out)
    np.clip(rgb_out, 0.0, None, out=rgb_out)
    return rgb_out
