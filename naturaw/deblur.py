"""Estimation aveugle du flou de bouge et deconvolution.

Le flou de bouge d'un appareil tenu a la main est, en premiere approximation,
une convolution par un noyau unique (la trajectoire du capteur pendant
l'exposition). On ne connait ni cette trajectoire ni l'image nette : c'est un
probleme *aveugle*, traite ici en trois temps.

1. **Estimation du noyau** (``estimate_motion_kernel``) : minimisation alternee
   dans le domaine des gradients, du grossier vers le fin (approche Cho & Lee /
   Xu & Jia simplifiee). On alterne une estimation de l'image latente (Wiener
   avec a priori de gradient, puis selection des contours les plus francs) et
   une estimation du noyau au sens des moindres carres, sous contraintes de
   positivite, de parcimonie, de support connexe et de recentrage.
   L'estimation se fait sur des imagettes *fenetrees* (Hann) prises dans les
   zones les plus texturees : sans ce fenetrage, le repliement circulaire de la
   FFT fausse completement le resultat (correlation du noyau 0.89 au lieu de
   1.00 sur un cas de reference).

2. **Validation** (``analyze_blur``) : l'estimateur a un plancher d'environ
   1.5 px, et il devient instable sur une image pauvre en texture. On ne
   deconvolue donc en mode ``auto`` que si l'etendue du noyau depasse
   nettement ce plancher, si le noyau est franchement directionnel (un flou
   isotrope est une mise au point ratee, que l'on ne cherche pas a corriger),
   et si deux imagettes eloignees donnent des noyaux compatibles.

3. **Deconvolution non aveugle** (``richardson_lucy``) : Richardson-Lucy
   (adapte au bruit de photons) regularise par variation totale, puis
   garde-fous anti-halo : la correction est bornee par l'intervalle local de
   l'image d'origine et appliquee en *rapport* de luminance, ce qui evite les
   liseres et les franges colorees d'une deconvolution brutale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .colorspace import luminance
from .filters import local_range

_BORDER = cv2.BORDER_REFLECT


@dataclass
class DeblurParams:
    """Reglages du deflou de bouge."""

    mode: str = "auto"
    """``off``, ``auto`` (n'agit que si un bouge franc est detecte), ``on``
    (ignore les tests d'anisotropie et de coherence, garde celui de nettete)
    ou ``force`` (deconvolue sans aucun test : a reserver aux cas ou l'on sait
    qu'il y a du bouge, de preference avec --deblur-longueur)."""

    strength: float = 0.8
    """Dose de correction appliquee (1 = deconvolution complete)."""

    iterations: int = 18
    """Iterations de Richardson-Lucy."""

    tv_weight: float = 0.003
    """Poids de la regularisation par variation totale."""

    kernel_size: int = 0
    """Taille du noyau estime ; 0 = automatique (31 px)."""

    crop_size: int = 512
    """Taille des imagettes d'analyse (pleine resolution)."""

    crops: int = 4
    """Nombre d'imagettes analysees conjointement pour estimer le noyau
    (elles servent aussi, par groupes disjoints, de test de stabilite)."""

    min_extent: float = 2.0
    """Etendue majeure minimale du noyau, en px, pour agir en mode auto."""

    min_anisotropy: float = 1.8
    """Rapport d'etendue majeure/mineure minimal (flou directionnel)."""

    confident_anisotropy: float = 2.6
    """Anisotropie a partir de laquelle la dose complete est appliquee.

    Entre ``min_anisotropy`` et cette valeur, la dose monte progressivement.
    L'asymetrie des risques l'impose : sur les captures d'essai, une
    deconvolution juste rapporte de 0,3 a 1,8 dB, une deconvolution fondee sur
    une estimation fausse en coute 6,8. Les cas francs (anisotropie 2,6 a 3,3)
    gardent donc toute la dose, les cas douteux n'en recoivent presque rien."""

    channels: str = "luma"
    """``luma`` (transfert de nettete, defaut) ou ``rgb`` (par canal)."""

    free_kernel: bool = False
    """Utiliser le noyau libre estime plutot que sa projection rectiligne.

    Par defaut le noyau estime est remplace par la trajectoire *rectiligne* de
    meme etendue et meme orientation. Un noyau libre estime sur quelques
    imagettes bruitees porte des lobes parasites qui font diverger la
    deconvolution ; une droite propre est un bien meilleur modele d'un bouge
    court. Mesure sur captures simulees : somme des gains de SSIM +0,42 avec
    la projection contre +0,11 avec le noyau libre, et jusqu'a 100 % du gain
    qu'aurait donne le vrai noyau."""

    length: float = 0.0
    """Flou rectiligne impose (px). > 0 : court-circuite l'estimation."""

    angle: float = 0.0
    """Angle du flou impose (degres, sens trigonometrique)."""


def _confidence(aniso: float, lo: float, hi: float) -> float:
    """Confiance [0,1] tiree de l'anisotropie du noyau estime."""
    if hi <= lo:
        return 1.0 if aniso >= hi else 0.0
    t = float(np.clip((aniso - lo) / (hi - lo), 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


@dataclass
class BlurEstimate:
    """Resultat de l'analyse de flou."""

    kernel: np.ndarray
    """Noyau retenu pour la deconvolution (projete, sauf demande contraire)."""
    extent: float = 0.0
    minor: float = 0.0
    anisotropy: float = 1.0
    angle: float = 0.0
    agreement: float = 1.0
    """Stabilite de l'estimation entre deux groupes d'imagettes disjoints."""
    reliable: bool = False
    reason: str = ""
    per_crop: List[Tuple[float, float]] = field(default_factory=list)
    free_kernel: Optional[np.ndarray] = None
    """Noyau libre tel qu'estime, avant projection (pour inspection)."""
    projected: bool = False

    def describe(self) -> str:
        shape = "trainee rectiligne" if self.projected else "noyau libre"
        return (f"{shape} {self.extent * np.sqrt(12.0):.0f} px a {self.angle:.0f} deg "
                f"(anisotropie {self.anisotropy:.1f}, stabilite {self.agreement:.2f})")


# --------------------------------------------------------------------------
# outils de base
# --------------------------------------------------------------------------

def _odd(n) -> int:
    n = int(max(3, round(n)))
    return n if n % 2 else n + 1


def _psf2otf(psf: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Transformee de Fourier d'un noyau centre, sans decalage de phase."""
    h, w = shape
    pad = np.zeros((h, w), dtype=np.float32)
    kh, kw = psf.shape
    pad[:kh, :kw] = psf
    pad = np.roll(pad, -(kh // 2), axis=0)
    pad = np.roll(pad, -(kw // 2), axis=1)
    return np.fft.rfft2(pad)


def convolve(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Convolution (et non correlation), adaptee a la taille du noyau."""
    k = np.ascontiguousarray(k, dtype=np.float32)
    if k.size <= 2048:
        return cv2.filter2D(img, -1, k[::-1, ::-1], borderType=_BORDER)
    from scipy.signal import oaconvolve
    ph, pw = k.shape[0] // 2, k.shape[1] // 2
    padded = cv2.copyMakeBorder(img, ph, k.shape[0] - 1 - ph, pw, k.shape[1] - 1 - pw, _BORDER)
    return oaconvolve(padded, k, mode="valid").astype(np.float32, copy=False)


def correlate(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Correlation par ``k`` (= l'adjoint K^T de la convolution)."""
    return convolve(img, k[::-1, ::-1])


def kernel_moments(k: np.ndarray) -> Tuple[float, float, float, float]:
    """(etendue majeure, mineure, anisotropie, angle) du noyau, en px/degres."""
    s = float(k.sum())
    if s <= 0:
        return 0.0, 0.0, 1.0, 0.0
    kk = (k / s).astype(np.float64)
    yy, xx = np.mgrid[0:k.shape[0], 0:k.shape[1]].astype(np.float64)
    cy = float((kk * yy).sum())
    cx = float((kk * xx).sum())
    vyy = float((kk * (yy - cy) ** 2).sum())
    vxx = float((kk * (xx - cx) ** 2).sum())
    vxy = float((kk * (yy - cy) * (xx - cx)).sum())
    m = np.array([[vyy, vxy], [vxy, vxx]])
    vals, vecs = np.linalg.eigh(m)
    lmin = max(float(vals[0]), 1e-6)
    lmax = max(float(vals[1]), 1e-6)
    v = vecs[:, 1]
    angle = float(np.degrees(np.arctan2(-v[0], v[1])) % 180.0)
    return float(np.sqrt(lmax)), float(np.sqrt(lmin)), float(np.sqrt(lmax / lmin)), angle


def kernel_spread(k: np.ndarray) -> float:
    """Rayon quadratique moyen du noyau autour de son barycentre."""
    major, minor, _, _ = kernel_moments(k)
    return float(np.sqrt(major * major + minor * minor))


def delta_kernel(size: int = 3) -> np.ndarray:
    k = np.zeros((_odd(size), _odd(size)), np.float32)
    k[k.shape[0] // 2, k.shape[1] // 2] = 1.0
    return k


def normalize_kernel(k: np.ndarray, prune: float = 0.03) -> np.ndarray:
    """Positivite, elagage, support connexe, recentrage, somme unitaire."""
    k = np.nan_to_num(np.array(k, dtype=np.float32, copy=True), nan=0.0,
                      posinf=0.0, neginf=0.0)
    np.clip(k, 0.0, None, out=k)
    peak = float(k.max())
    if peak <= 0:
        return delta_kernel(k.shape[0])
    if prune > 0:
        k[k < prune * peak] = 0.0
    # une seule composante connexe : les taches isolees sont des artefacts
    mask = (k > 0).astype(np.uint8)
    n, labels = cv2.connectedComponents(mask, connectivity=8)
    if n > 2:
        energies = [(float(k[labels == lab].sum()), lab) for lab in range(1, n)]
        keep = max(energies)[1]
        k[labels != keep] = 0.0
    total = float(k.sum())
    if total <= 0:
        return delta_kernel(k.shape[0])
    k /= total
    # recentrage sur le barycentre, sinon l'image entiere se decale
    yy, xx = np.mgrid[0:k.shape[0], 0:k.shape[1]].astype(np.float32)
    dy = int(round(k.shape[0] // 2 - float((k * yy).sum())))
    dx = int(round(k.shape[1] // 2 - float((k * xx).sum())))
    if dy or dx:
        k = np.roll(np.roll(k, dy, axis=0), dx, axis=1)
        s = float(k.sum())
        if s > 0:
            k /= s
    return k


def linear_motion_kernel(length: float, angle_deg: float, ksize: int = 0) -> np.ndarray:
    """Noyau de bouge rectiligne (mode manuel et tests)."""
    length = max(float(length), 1.0)
    if ksize <= 0:
        ksize = _odd(int(np.ceil(length)) + 2)
    ksize = _odd(ksize)
    k = np.zeros((ksize, ksize), np.float32)
    c = ksize // 2
    ang = np.deg2rad(angle_deg)
    for t in np.linspace(-length / 2.0, length / 2.0, int(max(8, length * 8))):
        yy = c - t * np.sin(ang)
        xx = c + t * np.cos(ang)
        y0, x0 = int(np.floor(yy)), int(np.floor(xx))
        fy, fx = yy - y0, xx - x0
        for dy, wy in ((0, 1.0 - fy), (1, fy)):
            for dx, wx in ((0, 1.0 - fx), (1, fx)):
                if 0 <= y0 + dy < ksize and 0 <= x0 + dx < ksize:
                    k[y0 + dy, x0 + dx] += wy * wx
    return normalize_kernel(k, prune=0.0)


# --------------------------------------------------------------------------
# preparation des imagettes d'analyse
# --------------------------------------------------------------------------

def _hann2(h: int, w: int) -> np.ndarray:
    return (np.hanning(h)[:, None] * np.hanning(w)[None, :]).astype(np.float32)


def prepare_crops(luma: np.ndarray, size: int = 640, count: int = 2):
    """Extrait les imagettes les plus texturees, normalisees et fenetrees.

    Renvoie une liste de ``(imagette, energie_texture)``. L'energie sert a
    refuser l'analyse sur une image trop pauvre (ciel uni, mur lisse) ou
    l'estimation du noyau n'est pas contrainte.

    Le choix des zones ne se fait pas sur la seule quantite de contours : une
    zone qui n'a *qu'une* orientation (une facade de fenetres alignees, des
    rayures) ne dit rien de l'etalement du noyau dans cette direction, et le
    probleme reste mal pose. On privilegie donc les zones a la fois contrastees
    et riches en orientations differentes, mesurees par le tenseur de
    structure local.
    """
    h, w = luma.shape[:2]
    size = int(min(size, h, w))
    size -= size % 2
    if size < 96:
        return []
    # Le choix des zones n'a pas besoin de la pleine resolution : on le fait
    # sur une version reduite (jusqu'a 16x moins de pixels a filtrer), ce qui
    # rend cette etape negligeable meme sur un fichier de 50 Mpx.
    scan_scale = 1.0
    scan = luma
    if max(h, w) > 1600:
        scan_scale = 1600.0 / max(h, w)
        scan = cv2.resize(luma, (max(64, int(round(w * scan_scale))),
                                 max(64, int(round(h * scan_scale)))),
                          interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(scan, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(scan, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    half = max(3, (int(round(size * scan_scale)) // 2) | 1)
    energy = cv2.boxFilter(grad, -1, (half, half), normalize=True,
                           borderType=cv2.BORDER_REFLECT)
    jxx = cv2.boxFilter(gx * gx, -1, (half, half), normalize=True, borderType=cv2.BORDER_REFLECT)
    jyy = cv2.boxFilter(gy * gy, -1, (half, half), normalize=True, borderType=cv2.BORDER_REFLECT)
    jxy = cv2.boxFilter(gx * gy, -1, (half, half), normalize=True, borderType=cv2.BORDER_REFLECT)
    coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / np.maximum(jxx + jyy, 1e-9)
    energy = energy * (1.0 - 0.7 * np.clip(coherence, 0.0, 1.0))
    if scan_scale != 1.0:
        energy = cv2.resize(energy, (w, h), interpolation=cv2.INTER_NEAREST)
    taken: List[Tuple[int, int]] = []
    crops = []
    work = energy.copy()
    for _ in range(max(1, int(count))):
        yy, xx = np.unravel_index(int(np.argmax(work)), work.shape)
        cy = int(np.clip(yy, size // 2, h - size // 2))
        cx = int(np.clip(xx, size // 2, w - size // 2))
        if any(abs(cy - py) < size // 2 and abs(cx - px) < size // 2 for py, px in taken):
            work[max(0, yy - size): yy + size, max(0, xx - size): xx + size] = -1.0
            if float(work.max()) <= 0:
                break
            continue
        taken.append((cy, cx))
        patch = luma[cy - size // 2: cy + size // 2, cx - size // 2: cx + size // 2]
        patch = np.ascontiguousarray(patch, dtype=np.float32)
        lo, hi = np.percentile(patch, (1.0, 99.0))
        norm = np.clip((patch - lo) / max(hi - lo, 1e-3), 0.0, 1.0).astype(np.float32)
        texture = float(np.mean(cv2.magnitude(
            cv2.Sobel(norm, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(norm, cv2.CV_32F, 0, 1, ksize=3))))
        crops.append(((norm - float(norm.mean())) * _hann2(*norm.shape), texture))
        work[max(0, cy - size): cy + size, max(0, cx - size): cx + size] = -1.0
        if float(work.max()) <= 0:
            break
    return crops


# --------------------------------------------------------------------------
# estimation aveugle
# --------------------------------------------------------------------------

def _latent_estimate(y: np.ndarray, k: np.ndarray, lam: float) -> np.ndarray:
    """Image latente : Wiener avec a priori de gradient L2 (resolu en Fourier)."""
    shape = y.shape
    K = _psf2otf(k, shape)
    Dx = _psf2otf(np.array([[1.0, -1.0]], np.float32), shape)
    Dy = _psf2otf(np.array([[1.0], [-1.0]], np.float32), shape)
    num = np.conj(K) * np.fft.rfft2(y)
    den = (K.conj() * K).real + lam * ((Dx.conj() * Dx).real + (Dy.conj() * Dy).real)
    return np.fft.irfft2(num / np.maximum(den, 1e-6), s=shape).astype(np.float32, copy=False)


def _threshold_gradients(x: np.ndarray, keep: float):
    """Ne garde que la fraction ``keep`` des gradients les plus forts.

    C'est ce qui rend l'estimation robuste : seuls les contours francs
    portent une information fiable sur la trajectoire ; les micro-details et
    le bruit n'apporteraient que des solutions parasites.
    """
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    keep = float(np.clip(keep, 0.005, 1.0))
    thr = float(np.quantile(mag, 1.0 - keep))
    mask = mag >= max(thr, 1e-6)
    return (gx * mask).astype(np.float32), (gy * mask).astype(np.float32)


def _kernel_from_gradients(pairs, ksize: int, gamma: float, keep: float) -> np.ndarray:
    """``min_k Σ_z Σ_d ||k * ∂_d x_z - ∂_d y_z||² + γ||k||²`` resolu en Fourier.

    La somme porte sur *plusieurs imagettes* z : le noyau de bouge est commun a
    toute l'image, alors qu'une imagette isolee peut etre degeneree. Une zone
    qui ne contient que des contours horizontaux, par exemple, ne contraint pas
    du tout l'etalement horizontal du noyau ; une autre zone, avec d'autres
    orientations, complete l'information. Estimer conjointement, c'est resoudre
    le vrai probleme au lieu de moyenner des reponses partielles.
    """
    shape = pairs[0][0].shape
    num = np.zeros((shape[0], shape[1] // 2 + 1), np.complex128)
    den = np.zeros((shape[0], shape[1] // 2 + 1), np.float64)
    for y, x in pairs:
        gxy = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
        gyy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
        gxx, gyx = _threshold_gradients(x, keep)
        for xs, ys in ((gxx, gxy), (gyx, gyy)):
            X = np.fft.rfft2(xs)
            num += np.conj(X) * np.fft.rfft2(ys)
            den += (X.conj() * X).real
    k_full = np.fft.fftshift(np.fft.irfft2(num / np.maximum(den + gamma, 1e-8), s=shape))
    cy, cx, h = shape[0] // 2, shape[1] // 2, ksize // 2
    k = k_full[cy - h: cy + h + 1, cx - h: cx + h + 1]
    return normalize_kernel(np.ascontiguousarray(k, dtype=np.float32), prune=0.02)


def estimate_motion_kernel(crops, kernel_size: int = 31, iterations: int = 6,
                           levels: int = 0, init: Optional[np.ndarray] = None
                           ) -> np.ndarray:
    """Estime le noyau de bouge sur une ou plusieurs imagettes fenetrees.

    Minimisation alternee du grossier vers le fin : a chaque echelle on estime
    l'image latente de chaque imagette avec le noyau courant, puis le noyau
    conjointement sur toutes les imagettes.

    ``init`` fournit un noyau de depart deja resolu : les echelles grossieres
    sont alors inutiles et l'on ne raffine qu'a pleine resolution. C'est ce
    qui rend le test de stabilite abordable (six fois moins de calcul).
    """
    if isinstance(crops, np.ndarray):
        crops = [crops]
    crops = [np.ascontiguousarray(c, dtype=np.float32) for c in crops]
    ksize = _odd(kernel_size)
    if levels <= 0:
        levels = 1 if init is not None else max(
            1, int(np.ceil(np.log2(max(ksize, 5) / 5.0))) + 1)
    k: Optional[np.ndarray] = normalize_kernel(
        cv2.resize(init, (ksize, ksize), interpolation=cv2.INTER_CUBIC), prune=0.02
    ) if init is not None else None
    for lvl in range(levels):
        scale = 0.5 ** (levels - 1 - lvl)
        ks = _odd(max(5, round(ksize * scale)))
        if scale < 1.0:
            level_crops = [cv2.resize(c, (max(64, int(round(c.shape[1] * scale))),
                                          max(64, int(round(c.shape[0] * scale)))),
                                      interpolation=cv2.INTER_AREA) for c in crops]
        else:
            level_crops = crops
        if k is None:
            k = delta_kernel(ks)
        elif k.shape[0] != ks:
            k = normalize_kernel(cv2.resize(k, (ks, ks), interpolation=cv2.INTER_CUBIC),
                                 prune=0.02)
        lam = 4e-3
        for it in range(iterations):
            keep = 0.18 * (0.8 ** it) + 0.02
            gamma = 1.0 * (0.7 ** it) + 0.02
            pairs = []
            for yl in level_crops:
                x = _latent_estimate(yl, k, lam)
                x = cv2.bilateralFilter(x, 5, 0.04, 3)
                pairs.append((yl, x))
            k_new = _kernel_from_gradients(pairs, ks, gamma=gamma, keep=keep)
            k = normalize_kernel(0.4 * k + 0.6 * k_new if it else k_new, prune=0.02)
            lam = max(lam * 0.7, 3e-4)
    return normalize_kernel(k, prune=0.05)


def analyze_blur(luma: np.ndarray, params: DeblurParams = DeblurParams(),
                 relaxed: bool = False) -> BlurEstimate:
    """Estime le noyau de bouge et dit s'il est exploitable.

    ``relaxed`` (mode ``on``) ne garde que le test de nettete : l'utilisateur
    affirme que la photo est floue, on ne discute que du cas ou aucun flou
    n'est mesurable.
    """
    if params.length > 0.0:
        k = linear_motion_kernel(params.length, params.angle)
        major, minor, aniso, angle = kernel_moments(k)
        est = BlurEstimate(k, major, minor, aniso, angle, 1.0, True, "noyau impose")
        est.projected = True
        return est

    ksize = _odd(params.kernel_size or 31)
    crops = prepare_crops(luma, size=params.crop_size, count=params.crops)
    if not crops:
        return BlurEstimate(delta_kernel(), reason="image trop petite pour l'analyse")

    textures = [t for _c, t in crops]
    patches = [c for c, _t in crops]

    def finish(estimate: BlurEstimate) -> BlurEstimate:
        """Remplace le noyau libre par la droite de meme etendue et angle."""
        if params.free_kernel or estimate.extent <= 0.3:
            return estimate
        estimate.free_kernel = estimate.kernel
        length = estimate.extent * np.sqrt(12.0)   # ecart-type d'un segment = L/sqrt(12)
        estimate.kernel = linear_motion_kernel(length, estimate.angle)
        estimate.projected = True
        return estimate

    # Pre-test rapide : une seule imagette, un noyau plus petit et moins
    # d'iterations. L'estimation sur une zone isolee a tendance a *sur*-estimer
    # le flou ; si meme elle ne voit rien, l'image est nette et on s'arrete la.
    # La plupart des photos passent par ce chemin, qui coute quatre fois moins.
    if len(patches) > 1 and params.length <= 0.0:
        quick = estimate_motion_kernel(patches[:1], kernel_size=min(ksize, 21),
                                       iterations=4)
        q_major, q_minor, q_aniso, q_angle = kernel_moments(quick)
        if q_major < params.min_extent * 0.8:
            est = BlurEstimate(quick, q_major, q_minor, q_aniso, q_angle)
            est.reason = f"image deja nette (etendue {q_major:.1f} px)"
            return est

    k = estimate_motion_kernel(patches, kernel_size=ksize)
    major, minor, aniso, angle = kernel_moments(k)
    est = BlurEstimate(k, major, minor, aniso, angle)
    est.agreement = 1.0
    est.per_crop = [(major, angle)]

    # 1. assez de texture pour contraindre le probleme ?
    if max(textures) < 0.02:
        est.reason = "pas assez de texture pour estimer le bouge"
        return est
    # 2. flou franc ? (l'estimateur a un plancher d'environ 1.5 px)
    if major < params.min_extent:
        est.reason = f"image deja nette (etendue {major:.1f} px)"
        return est
    # 3. flou directionnel ? un flou isotrope est une erreur de mise au point
    if not relaxed and aniso < params.min_anisotropy:
        est.reason = (f"flou non directionnel (anisotropie {aniso:.1f}) : "
                      "probable erreur de mise au point, non traitee")
        return est
    # 4. estimation stable ? Sur certaines scenes (grandes surfaces lisses,
    #    motifs periodiques comme des rangees de fenetres) le probleme aveugle
    #    reste mal pose : deux groupes d'imagettes disjoints donnent alors des
    #    noyaux tres differents. Dans ce cas on refuse de deconvoluer plutot
    #    que d'appliquer un noyau faux, qui creerait des artefacts.
    if len(patches) >= 2 and not relaxed:
        # Les deux groupes repartent de zero, et surtout pas du noyau conjoint :
        # les faire demarrer du meme point les ferait converger ensemble, et le
        # test de stabilite ne testerait plus rien (essaye : une image nette de
        # 6 Mpx passait alors pour floue de 16 px).
        k_a = estimate_motion_kernel(patches[0::2], kernel_size=ksize)
        k_b = estimate_motion_kernel(patches[1::2], kernel_size=ksize)
        ma, _na, _aa, aa = kernel_moments(k_a)
        mb, _nb, _ab, ab = kernel_moments(k_b)
        est.agreement = float(min(ma, mb) / max(max(ma, mb), 1e-6))
        d_ang = abs(aa - ab) % 180.0
        d_ang = min(d_ang, 180.0 - d_ang)
        est.per_crop = [(ma, aa), (mb, ab)]
        if est.agreement < 0.6 or d_ang > 30.0:
            est.reason = (f"estimation instable sur cette scene "
                          f"(etendues {ma:.1f} et {mb:.1f} px, ecart d'angle "
                          f"{d_ang:.0f} deg) : deflou non applique")
            return est
    elif not relaxed and aniso < 2.2:
        # une seule zone exploitable : on n'a aucun moyen de verifier
        # l'estimation, donc on exige un bouge franchement directionnel
        est.reason = (f"une seule zone analysable et flou peu directionnel "
                      f"(anisotropie {aniso:.1f}) : deflou non applique")
        return est
    est.reliable = True
    est.reason = f"bouge detecte ({len(patches)} zones, accord {est.agreement:.2f})"
    return finish(est)


# --------------------------------------------------------------------------
# deconvolution non aveugle
# --------------------------------------------------------------------------

def _tv_divergence(x: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Divergence du champ de gradient normalise (terme de variation totale)."""
    gx = np.diff(x, axis=1, append=x[:, -1:])
    gy = np.diff(x, axis=0, append=x[-1:, :])
    mag = np.sqrt(gx * gx + gy * gy + eps * eps)
    gx /= mag
    gy /= mag
    return ((gx - np.roll(gx, 1, axis=1)) + (gy - np.roll(gy, 1, axis=0))
            ).astype(np.float32, copy=False)


def richardson_lucy(img: np.ndarray, k: np.ndarray, iterations: int = 18,
                    tv_weight: float = 0.003) -> np.ndarray:
    """Richardson-Lucy regularise par variation totale (image lineaire >= 0)."""
    y = np.clip(np.asarray(img, dtype=np.float32), 0.0, None)
    scale = max(float(np.percentile(y, 99.9)), 1e-4)
    y = y / scale
    x = y.copy()
    for _ in range(max(1, int(iterations))):
        est = convolve(x, k)
        np.maximum(est, 1e-6, out=est)
        corr = correlate(y / est, k)
        if tv_weight > 0.0:
            reg = 1.0 - tv_weight * _tv_divergence(x)
            np.clip(reg, 0.5, 1.5, out=reg)
            corr /= reg
        x *= corr
        np.clip(x, 0.0, 4.0, out=x)
    return (x * scale).astype(np.float32, copy=False)


def _deringed(deconv: np.ndarray, original: np.ndarray, sigma: float,
              radius: int = 2, margin: float = 0.08) -> np.ndarray:
    """Borne la deconvolution a l'intervalle local d'origine (anti-halo)."""
    lo, hi = local_range(original, radius)
    slack = margin * (hi - lo) + 2.0 * float(sigma)
    return np.clip(deconv, lo - slack, hi + slack)


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------

def deblur_linear_rgb(
    rgb: np.ndarray,
    params: DeblurParams = DeblurParams(),
    noise_sigma: float = 0.002,
    report: Optional[dict] = None,
    estimate: Optional[BlurEstimate] = None,
) -> np.ndarray:
    """Retire le flou de bouge d'une image RGB lineaire.

    En mode ``auto``, l'image est renvoyee inchangee si aucun bouge franc et
    coherent n'est detecte. Par defaut la correction est calculee sur la
    luminance puis reportee sur les trois canaux *en rapport*, ce qui laisse
    teinte et saturation intactes (pas de frange coloree).
    """
    mode = (params.mode or "auto").lower()
    if mode in ("off", "non", "none", "0") or params.strength <= 0.0:
        return rgb
    forced = mode in ("force", "forcer")
    rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    y_full = luminance(rgb)

    if estimate is None:
        estimate = analyze_blur(y_full, params, relaxed=mode in ("on", "oui"))
        if forced and not estimate.reliable:
            estimate.reliable = True
            if not estimate.projected and not params.free_kernel and estimate.extent > 0.3:
                estimate.free_kernel = estimate.kernel
                estimate.kernel = linear_motion_kernel(estimate.extent * np.sqrt(12.0),
                                                       estimate.angle)
                estimate.projected = True
    if report is not None:
        report["deblur_analysis"] = estimate.describe()
        report["deblur_reason"] = estimate.reason
    if not estimate.reliable:
        if report is not None:
            report["deblur"] = f"ignore : {estimate.reason}"
        return rgb
    kernel = estimate.kernel
    if kernel_moments(kernel)[0] < 0.6:  # noyau degenere : rien a faire
        if report is not None:
            report["deblur"] = "ignore : noyau degenere"
        return rgb

    strength = float(np.clip(params.strength, 0.0, 1.0))
    confidence = 1.0
    if estimate.reason != "noyau impose" and params.length <= 0.0:
        confidence = _confidence(estimate.anisotropy, params.min_anisotropy,
                                 params.confident_anisotropy)
        strength *= confidence
    if strength < 0.05:
        if report is not None:
            report["deblur"] = (f"ignore : estimation trop incertaine "
                                f"(anisotropie {estimate.anisotropy:.1f})")
        return rgb
    if report is not None:
        report["deblur"] = (f"noyau {kernel.shape[0]} px, {estimate.describe()}, "
                            f"dose {strength:.2f}"
                            + (f" (confiance {confidence:.2f})" if confidence < 0.99 else "")
                            + f", {params.channels}")

    if params.channels == "rgb":
        out = np.empty_like(rgb)
        for c in range(3):
            dec = richardson_lucy(rgb[..., c], kernel, params.iterations, params.tv_weight)
            dec = _deringed(dec, rgb[..., c], noise_sigma)
            out[..., c] = rgb[..., c] + strength * (dec - rgb[..., c])
        np.clip(out, 0.0, None, out=out)
        return out

    dec = richardson_lucy(y_full, kernel, params.iterations, params.tv_weight)
    dec = _deringed(dec, y_full, noise_sigma)
    target = y_full + strength * (dec - y_full)
    eps = 2e-3
    factor = (target + eps) / (y_full + eps)
    np.clip(factor, 0.25, 4.0, out=factor)
    out = rgb * factor[..., None]
    np.clip(out, 0.0, None, out=out)
    return out
