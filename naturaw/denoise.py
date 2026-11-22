"""Debruitage multi-echelle "naturel".

Principe : plutot que de seuiller des coefficients (ce qui laisse des traces
en plaques et un rendu plastique), on applique a chaque echelle un *filtre de
Wiener local* : le gain conserve exactement la part de signal qui depasse le
bruit mesure dans le voisinage.

    gain = max(0, (E - var_bruit) / E)        avec E = energie locale moyenne

Trois choix font le rendu naturel :

1. **Luminance et chrominance separees.** Le bruit chroma (plaques de couleur)
   est beaucoup plus visible et porte beaucoup moins d'information : on le
   traite deux fois plus fort, puis on le nettoie avec un filtre guide par la
   luminance. La luminance, elle, est a peine touchee -> la nettete subjective
   est conservee.
2. **Protection des zones texturees.** Un masque de detail (contraste local
   rapporte au bruit local) reinjecte une part des coefficients dans les
   feuillages, tissus, cheveux... la ou un debruiteur classique "efface".
3. **Grain conserve.** Une fraction du bruit fin retire est remise dans la
   luminance seulement. Une image totalement lisse se *voit* ; un grain fin
   residuel, achromatique, est ce qui rend le resultat credible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import cv2
import numpy as np

from .colorspace import linear_to_srgb, rgb_to_ycocg, ycocg_to_rgb
from .filters import box, cross_bilateral, starlet_smooth
from .noise import NoiseProfile, ScaleNoise, measure_profile

# Rayon de la fenetre d'estimation d'energie, par echelle.
_WINDOW = (2, 3, 5, 8, 10, 12, 14, 16)
# Ponderation du seuil par echelle (legerement plus ferme sur les grandes
# echelles, ou le bruit basse frequence se voit en plaques).
_SCALE_WEIGHT = (1.0, 1.0, 1.05, 1.12, 1.18, 1.25, 1.3, 1.3)
_EPS = 1e-12


@dataclass
class DenoiseParams:
    """Reglages du debruitage (tous relatifs : 1.0 = reglage nominal)."""

    strength: float = 1.0
    """Force globale. 0 desactive, 0.6 = discret, 1.6 = fort."""

    chroma: float = 1.5
    """Multiplicateur de force sur la chrominance.

    Valeur calee par mesure : au-dela, la couleur des textures fines commence
    a etre lissee pour un gain de bruit negligeable."""

    grain: float = 0.16
    """Fraction du grain fin de luminance conservee (0 = lisse, 0.3 = argentique)."""

    detail: float = 0.45
    """Protection des zones texturees (0 = aucune, 1 = maximale)."""

    levels: int = 0
    """Nombre d'echelles ; 0 = automatique selon la taille de l'image."""

    chroma_cleanup: float = 0.0
    """Passe supplementaire contre les plaques de couleur (0 = desactivee).

    Le Wiener multi-echelle traite deja le bruit chroma basse frequence, parce
    que le bruit est mesure echelle par echelle. Cette passe n'est utile que
    sur des fichiers ou des taches de couleur subsistent malgre tout ; mise
    par defaut, elle biaisait legerement les degrades propres (un ciel)."""

    safety: float = 5.0
    """Ecart maximal autorise par canal, en ecarts-types de bruit (0 = aucun)."""

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
}


def auto_levels(shape: Sequence[int], requested: int = 0) -> int:
    """Nombre d'echelles starlet adapte a la taille de l'image."""
    if requested and requested > 0:
        return int(min(requested, len(_WINDOW)))
    long_side = max(shape[0], shape[1])
    if long_side < 400:
        return 3
    if long_side < 1200:
        return 4
    if long_side < 3000:
        return 5
    return 6


def _detail_mask(luma: np.ndarray, sigma_map: np.ndarray) -> np.ndarray:
    """Masque [0,1] des zones ou le contraste local domine clairement le bruit."""
    mean = box(luma, 2)
    sq = box(luma * luma, 2)
    var = np.maximum(sq - mean * mean, 0.0)
    sd = np.sqrt(var, out=var)
    ratio = sd / np.maximum(sigma_map, 1e-6)
    # rampe lisse : 1.8 sigma -> rien, 6 sigma -> protection totale
    t = np.clip((ratio - 1.8) / 4.2, 0.0, 1.0)
    t *= t * (3.0 - 2.0 * t)
    return t.astype(np.float32, copy=False)


def _structure_mask(channel: np.ndarray, sigma_map: np.ndarray,
                    lo: float = 3.0, hi: float = 8.0, radius: int = 2) -> np.ndarray:
    """Masque [0,1] la ou la variation locale du canal domine son bruit.

    Applique a la chrominance, c'est ce qui protege les vrais aplats satures :
    sans lui, un debruiteur multi-echelle fait baver la couleur d'une zone
    saturee sur ses voisines sombres (la compensation entre echelles est
    rompue par l'attenuation).
    """
    mean = box(channel, radius)
    sq = box(channel * channel, radius)
    var = np.maximum(sq - mean * mean, 0.0)
    sd = np.sqrt(var, out=var)
    ratio = sd / np.maximum(sigma_map, 1e-6)
    t = np.clip((ratio - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    t *= t * (3.0 - 2.0 * t)
    return t.astype(np.float32, copy=False)


def _denoise_channel(
    channel: np.ndarray,
    brightness: np.ndarray,
    models: List[ScaleNoise],
    levels: int,
    strength: float,
    grain: float = 0.0,
    detail_mask: Optional[np.ndarray] = None,
    detail_amount: float = 0.0,
) -> np.ndarray:
    """Wiener local multi-echelle sur un canal."""
    out = np.zeros_like(channel, dtype=np.float32)
    cur = channel.astype(np.float32, copy=False)
    for j in range(levels):
        nxt = starlet_smooth(cur, j)
        w = cur - nxt
        k = strength * _SCALE_WEIGHT[min(j, len(_SCALE_WEIGHT) - 1)]
        var = models[j].variance(brightness) * (k * k)
        radius = _WINDOW[min(j, len(_WINDOW) - 1)]
        energy = box(w * w, radius)
        gain = (energy - var) / np.maximum(energy, _EPS)
        np.clip(gain, 0.0, 1.0, out=gain)
        # la carte de gain est elle-meme bruitee : on la lisse pour eviter
        # des plaques de nettete/bruit alternees
        gain = box(gain, radius)
        if detail_mask is not None and detail_amount > 0.0 and j <= 3:
            gain += detail_amount * (1.0 - gain) * detail_mask
            np.clip(gain, 0.0, 1.0, out=gain)
        kept = w * gain
        if grain > 0.0 and j <= 1:
            # grain fin remis a dose decroissante avec l'echelle
            g = grain if j == 0 else grain * 0.45
            kept += g * (w - kept)
        out += kept
        cur = nxt
    out += cur
    return out


def _chroma_cleanup(ycc: np.ndarray, profile: NoiseProfile, amount: float) -> None:
    """Supprime les plaques de couleur basse frequence restantes (sur place).

    Ce qui reste visible apres le Wiener multi-echelle, ce sont de grandes
    taches de couleur tres douces. On les traite la ou elles vivent : sur une
    version reduite de la chrominance, avec un bilateral joint guide par la
    luminance (donc sans bavure sur les contours). Seule la partie *basse
    frequence* de la chrominance est modifiee, ce qui garantit qu'aucun detail
    ni aucune couleur ne sont inventes.
    """
    if amount <= 0.0:
        return
    h, w = ycc.shape[:2]
    factor = 4 if min(h, w) >= 480 else 2
    if min(h, w) < 120:
        return
    small = (max(1, w // factor), max(1, h // factor))
    # Le guide est encode en gamma : l'ecart de luminance qui compte
    # perceptuellement est relatif, et le bruit y devient a peu pres uniforme
    # sur toute la plage tonale. Un seuil absolu en lineaire laisse fuir les
    # couleurs saturees dans les ombres voisines.
    guide_lo = cv2.resize(linear_to_srgb(np.clip(ycc[..., 0], 0.0, 1.0)), small,
                          interpolation=cv2.INTER_AREA)
    sigma_y = max(profile.channels[0][0].sigma_at(0.18), 2e-3)
    sigma_range = float(np.clip(5.0 * sigma_y * 1.2, 0.04, 0.25))
    brightness = cv2.resize(np.clip(ycc[..., 0], 0.0, None), small,
                            interpolation=cv2.INTER_AREA)
    brightness = cv2.resize(brightness, (w, h), interpolation=cv2.INTER_LINEAR)
    for c in (1, 2):
        chroma = ycc[..., c]
        lo = cv2.resize(chroma, small, interpolation=cv2.INTER_AREA)
        sm = cross_bilateral(guide_lo, lo, radius=3, sigma_range=sigma_range)
        # interpolation lineaire (et non cubique) : pas de lobe negatif, donc
        # pas de rebond de couleur au bord des aplats satures
        base = cv2.resize(lo, (w, h), interpolation=cv2.INTER_LINEAR)
        back = cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR)
        delta = back - base
        delta *= amount
        # Ce nettoyage ne doit retirer que du *bruit*, et il agit sur les
        # basses frequences : sa correction est donc bornee par le bruit
        # mesure *a cette echelle-la*, pas par le bruit fin (dix fois plus
        # grand). Sans cela, il "recentre" aussi les degrades propres -- un
        # ciel sans bruit ressortait legerement biaise.
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
    """Debruite une image RGB lineaire float32 dans [0, ~1].

    ``profile`` permet de reutiliser un profil de bruit deja mesure (p.ex.
    mesure sur l'image pleine resolution et applique a une tuile).
    """
    if params.strength <= 0.0:
        return rgb
    rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    levels = auto_levels(rgb.shape, params.levels)
    ycc = rgb_to_ycocg(rgb)
    if profile is None:
        profile = measure_profile(ycc, levels)
    if report is not None:
        report["noise_levels"] = levels
        report["noise_sigma"] = profile.sigma_report()
        report["noise_luma_sigma"] = profile.luma_sigma()

    brightness = box(ycc[..., 0], 3)
    np.clip(brightness, 0.0, None, out=brightness)

    sigma_map = np.sqrt(profile.channels[0][0].variance(brightness))
    mask = _detail_mask(ycc[..., 0], sigma_map) if params.detail > 0 else None
    del sigma_map

    out = np.empty_like(ycc)
    out[..., 0] = _denoise_channel(
        ycc[..., 0], brightness, profile.channels[0], levels,
        strength=params.strength, grain=params.grain,
        detail_mask=mask, detail_amount=params.detail,
    )
    chroma_strength = params.strength * params.chroma
    for c in (1, 2):
        sigma_c = np.sqrt(profile.channels[c][0].variance(brightness))
        # protection des aplats de couleur reels (voir _structure_mask)
        cmask = _structure_mask(ycc[..., c], sigma_c)
        if mask is not None:
            np.maximum(cmask, 0.3 * mask, out=cmask)
        del sigma_c
        out[..., c] = _denoise_channel(
            ycc[..., c], brightness, profile.channels[c], levels,
            strength=chroma_strength, grain=0.0,
            detail_mask=cmask, detail_amount=0.9 * max(params.detail, 0.2),
        )
        del cmask
    del mask

    _chroma_cleanup(out, profile, float(np.clip(params.chroma_cleanup, 0.0, 1.0)))

    if params.safety > 0.0:
        # Filet de securite, applique en dernier : aucun canal ne s'ecarte de
        # l'original de plus de ``safety`` ecarts-types de bruit. Un debruiteur
        # multi-echelle peut sinon, au voisinage d'un aplat tres sature,
        # reconstituer une couleur franche a partir d'une echelle grossiere
        # (defaut classique, visible sous forme de bavure de couleur).
        for c in range(3):
            limit = params.safety * np.sqrt(profile.channels[c][0].variance(brightness))
            np.clip(out[..., c], ycc[..., c] - limit, ycc[..., c] + limit,
                    out=out[..., c])
    del ycc


    rgb_out = ycocg_to_rgb(out)
    np.clip(rgb_out, 0.0, None, out=rgb_out)
    return rgb_out
