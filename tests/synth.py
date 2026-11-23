"""Fabrication d'images et de RAW synthetiques pour les tests.

Aucun RAW d'appareil n'est necessaire : on construit un DNG Bayer valide
(lisible par LibRaw) a partir d'une image de reference, ce qui permet de
verifier la chaine complete, du dematricage a l'ecriture, et de mesurer la
qualite par rapport a une verite terrain connue.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


# ---------------------------------------------------------------- metriques
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return float(10.0 * np.log10(1.0 / max(float(np.mean((a - b) ** 2)), 1e-12)))


def ssim(a: np.ndarray, b: np.ndarray, sigma: float = 1.5) -> float:
    """SSIM global sur une image monocanal (fenetre gaussienne)."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu1 = cv2.GaussianBlur(a, (0, 0), sigma)
    mu2 = cv2.GaussianBlur(b, (0, 0), sigma)
    s11 = cv2.GaussianBlur(a * a, (0, 0), sigma) - mu1 * mu1
    s22 = cv2.GaussianBlur(b * b, (0, 0), sigma) - mu2 * mu2
    s12 = cv2.GaussianBlur(a * b, (0, 0), sigma) - mu1 * mu2
    num = (2 * mu1 * mu2 + c1) * (2 * s12 + c2)
    den = (mu1 ** 2 + mu2 ** 2 + c1) * (s11 + s22 + c2)
    return float(np.mean(num / den))


def texture_mtf(out: np.ndarray, ref: np.ndarray, bands=(0.05, 0.10, 0.20, 0.35)):
    """Conservation de la texture, par bande de frequence (methode du spectre croise).

    Sur une mire "dead leaves", le rapport entre le spectre croise de l'image
    traitee et le spectre de la reference donne la part du *signal* conservee
    a chaque frequence. Contrairement a un simple rapport de spectres, il
    ignore le bruit residuel : une image bruitee n'obtient donc pas une bonne
    note en gardant du bruit a la place de la texture.

    1,0 = texture intacte ; 0,5 = la moitie du detail de cette finesse a
    disparu. Au-dessus de 1,0, l'image est accentuee.
    """
    a = out.mean(axis=2) if out.ndim == 3 else out
    b = ref.mean(axis=2) if ref.ndim == 3 else ref
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    a = a[:h, :w].astype(np.float64)
    b = b[:h, :w].astype(np.float64)
    win = np.outer(np.hanning(h), np.hanning(w))
    fa = np.fft.rfft2((a - a.mean()) * win)
    fb = np.fft.rfft2((b - b.mean()) * win)
    cross = np.real(fa * np.conj(fb))
    power = np.abs(fb) ** 2
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    rho = np.sqrt(fy ** 2 + fx ** 2)
    out_vals = {}
    for f in bands:
        sel = (rho >= f * 0.8) & (rho < f * 1.25)
        if np.count_nonzero(sel) < 8:
            out_vals[f] = float("nan")
            continue
        out_vals[f] = float(cross[sel].sum() / max(power[sel].sum(), 1e-12))
    return out_vals


def gradient_energy(img: np.ndarray) -> float:
    return float(np.mean(cv2.magnitude(cv2.Sobel(img, cv2.CV_32F, 1, 0),
                                       cv2.Sobel(img, cv2.CV_32F, 0, 1))))


# ------------------------------------------------------------------ scenes
def texture(h: int, w: int, seed: int = 0) -> np.ndarray:
    """Texture fractale a aretes (feuillage, tissu, gravier...)."""
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    for cell, amp in ((4, 0.5), (8, 0.4), (16, 0.35), (32, 0.3), (64, 0.25), (128, 0.2)):
        small = rng.random((h // cell + 2, w // cell + 2)).astype(np.float32)
        out += amp * cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    out -= out.min()
    return (out / max(out.max(), 1e-6)).astype(np.float32)


def architecture(h: int = 900, w: int = 1350, seed: int = 2) -> np.ndarray:
    """Scene structuree : horizon, facades, fenetres (pour le redressement)."""
    img = np.empty((h, w, 3), np.float32)
    sky = np.linspace(0.62, 0.38, h, dtype=np.float32)[:, None]
    img[...] = np.stack([sky * 0.55 + 0.04, sky * 0.72 + 0.04, sky + 0.05], -1)
    ground = int(h * 0.56)
    tex = texture(h - ground, w, seed + 1)
    img[ground:] = np.stack([0.10 + 0.14 * tex, 0.11 + 0.13 * tex, 0.09 + 0.11 * tex], -1)
    cv2.line(img, (0, ground), (w, ground), (0.72, 0.70, 0.66), 3)
    for i in range(6):
        x0 = int(w * 0.06 + i * w * 0.15)
        x1 = x0 + int(w * 0.105)
        y0 = int(h * (0.16 + 0.03 * (i % 3)))
        shade = 0.30 + 0.12 * ((i % 2) * 2 - 0.5)
        cv2.rectangle(img, (x0, y0), (x1, ground), (shade, shade * 0.96, shade * 0.9), -1)
        for j in range(5):
            wy = y0 + int(h * 0.05) + j * int(h * 0.075)
            if wy + int(h * 0.045) >= ground:
                break
            cv2.rectangle(img, (x0 + 6, wy), (x1 - 6, wy + int(h * 0.045)),
                          (0.05, 0.06, 0.09), -1)
    for i in range(5):  # elements colores satures
        x0 = int(w * 0.1 + i * w * 0.17)
        col = [(0.45, 0.04, 0.04), (0.04, 0.28, 0.06), (0.05, 0.07, 0.42),
               (0.44, 0.34, 0.03), (0.35, 0.05, 0.30)][i]
        cv2.rectangle(img, (x0, ground + int(h * 0.06)),
                      (x0 + int(w * 0.07), ground + int(h * 0.13)), col, -1)
    return np.clip(img, 0.004, 1.0).astype(np.float32)


def dead_leaves(h: int = 800, w: int = 1200, seed: int = 0, rmin: float = 2.0,
                rmax: float = 0.25, count: int = 0, colour: bool = True) -> np.ndarray:
    """Mire "dead leaves" : disques empiles de rayons en loi de puissance.

    C'est le modele de reference des statistiques d'images naturelles, et la
    mire que l'industrie photo utilise pour mesurer la *conservation de la
    texture* d'un debruiteur (norme ISO 19567-2) : son spectre en 1/f^2, ses
    contours francs a toutes les echelles et ses gradients a queue lourde
    reproduisent ce qui fait echouer les debruiteurs sur de vraies photos,
    ce qu'une scene geometrique ne fait pas.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.float32)
    r_max = rmax * min(h, w)
    if count <= 0:
        count = int(30 * (h * w) / (np.pi * rmin * r_max))
    # rayons en p(r) ~ r^-3 : la loi qui rend la mire invariante d'echelle
    u = rng.random(count).astype(np.float64)
    radii = 1.0 / np.sqrt(u / (rmin ** -2 - r_max ** -2) + r_max ** -2)
    xs = rng.integers(0, w, count)
    ys = rng.integers(0, h, count)
    if colour:
        cols = rng.random((count, 3)).astype(np.float32)
        # saturations variees mais realistes : on melange chaque teinte au gris
        mix = rng.random((count, 1)).astype(np.float32) ** 1.5
        grey = cols.mean(axis=1, keepdims=True)
        cols = grey + mix * (cols - grey)
    else:
        cols = np.repeat(rng.random((count, 1)).astype(np.float32), 3, axis=1)
    order = np.argsort(-radii)       # les grands disques d'abord
    for i in order:
        cv2.circle(img, (int(xs[i]), int(ys[i])), max(1, int(radii[i])),
                   tuple(float(v) for v in cols[i]), -1, lineType=cv2.LINE_AA)
    return np.clip(img * 0.9 + 0.03, 0.004, 1.0).astype(np.float32)


def simulate_raw_capture(scene: np.ndarray, full_well_e: float = 2500.0,
                         read_noise_e: float = 3.0, seed: int = 0,
                         black: float = 0.0):
    """Simule la prise de vue d'une scene par un capteur Bayer.

    Le bruit est ajoute *au niveau du photosite*, avant dematricage : bruit de
    photons (Poisson) et bruit de lecture (gaussien). C'est ce qui donne au
    bruit sa correlation spatiale et ses couleurs apres dematricage -- le
    bruit blanc ajoute apres coup, lui, ne ressemble a rien de reel.

    ``full_well_e`` correspond a la sensibilite : 40000 e- ~ 100 ISO,
    2500 e- ~ 1600 ISO, 600 e- ~ 6400 ISO, 150 e- ~ 25600 ISO.
    Renvoie la mosaique Bayer RGGB en [0, 1].
    """
    rng = np.random.default_rng(seed)
    h, w = scene.shape[:2]
    h -= h % 2
    w -= w % 2
    img = np.clip(scene[:h, :w], 0.0, 1.0)
    mosaic = np.zeros((h, w), np.float32)
    mosaic[0::2, 0::2] = img[0::2, 0::2, 0]
    mosaic[0::2, 1::2] = img[0::2, 1::2, 1]
    mosaic[1::2, 0::2] = img[1::2, 0::2, 1]
    mosaic[1::2, 1::2] = img[1::2, 1::2, 2]
    electrons = mosaic * float(full_well_e)
    noisy = rng.poisson(np.maximum(electrons, 0.0)).astype(np.float32)
    noisy += rng.normal(0.0, float(read_noise_e), noisy.shape).astype(np.float32)
    out = noisy / float(full_well_e) + float(black)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def write_dng_mosaic(path: str, mosaic: np.ndarray, white: int = 65535) -> str:
    """Ecrit une mosaique Bayer deja formee dans un DNG lisible par LibRaw."""
    import tifffile

    raw = np.clip(mosaic * white, 0, white).astype(np.uint16)
    extratags = [
        (50706, 1, 4, (1, 4, 0, 0), True),
        (50707, 1, 4, (1, 1, 0, 0), True),
        (33421, 3, 2, (2, 2), True),
        (33422, 1, 4, (0, 1, 1, 2), True),
        (50708, 2, None, "naturaw-synth", True),
        (50721, 10, 9, (1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1), True),
        (50728, 5, 3, (1, 1, 1, 1, 1, 1), True),
        (50717, 4, 1, (int(white),), True),
        (50714, 3, 1, (0,), True),
    ]
    tifffile.imwrite(path, raw, photometric=32803, planarconfig="contig",
                     extratags=extratags, compression=None)
    return path


def shake_kernel(size: int = 21, steps: int = 50, seed: int = 0,
                 scale: float = 1.0) -> np.ndarray:
    """Trajectoire de bouge aleatoire (marche lissee), normalisee."""
    from naturaw.deblur import normalize_kernel
    rng = np.random.default_rng(seed)
    k = np.zeros((size, size), np.float32)
    pos = np.array([size / 2.0, size / 2.0])
    vel = rng.normal(0, 0.6, 2)
    for _ in range(steps):
        vel = 0.85 * vel + rng.normal(0, 0.35, 2)
        pos = pos + vel * scale
        y, x = np.clip(pos, 0, size - 1.001)
        y0, x0 = int(y), int(x)
        fy, fx = y - y0, x - x0
        for dy, wy in ((0, 1 - fy), (1, fy)):
            for dx, wx in ((0, 1 - fx), (1, fx)):
                if 0 <= y0 + dy < size and 0 <= x0 + dx < size:
                    k[y0 + dy, x0 + dx] += wy * wx
    return normalize_kernel(cv2.GaussianBlur(k, (0, 0), 0.6), prune=0.0)


def add_photon_noise(rgb: np.ndarray, a: float = 1.8e-3, b: float = 1.0e-4,
                     seed: int = 7) -> np.ndarray:
    """Bruit realiste dependant du signal : var = a*I + b."""
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(np.maximum(a * rgb + b, 0.0)).astype(np.float32)
    out = rgb + rng.normal(0.0, 1.0, rgb.shape).astype(np.float32) * sigma
    return np.clip(out, 0.0, None).astype(np.float32)


# -------------------------------------------------------------------- DNG
def write_dng(path: str, rgb_linear: np.ndarray, white: int = 65535) -> str:
    """Ecrit un DNG Bayer RGGB (lisible par LibRaw) a partir d'une image RGB."""
    import tifffile

    h, w = rgb_linear.shape[:2]
    h -= h % 2
    w -= w % 2
    img = np.clip(rgb_linear[:h, :w], 0.0, 1.0)
    raw = np.zeros((h, w), np.uint16)
    raw[0::2, 0::2] = img[0::2, 0::2, 0] * white      # R
    raw[0::2, 1::2] = img[0::2, 1::2, 1] * white      # G
    raw[1::2, 0::2] = img[1::2, 0::2, 1] * white      # G
    raw[1::2, 1::2] = img[1::2, 1::2, 2] * white      # B
    extratags = [
        (50706, 1, 4, (1, 4, 0, 0), True),            # DNGVersion
        (50707, 1, 4, (1, 1, 0, 0), True),            # DNGBackwardVersion
        (33421, 3, 2, (2, 2), True),                  # CFARepeatPatternDim
        (33422, 1, 4, (0, 1, 1, 2), True),            # CFAPattern = RGGB
        (50708, 2, None, "naturaw-synth", True),      # UniqueCameraModel
        (50721, 10, 9, (1, 1, 0, 1, 0, 1,
                        0, 1, 1, 1, 0, 1,
                        0, 1, 0, 1, 1, 1), True),     # ColorMatrix1 = identite
        (50728, 5, 3, (1, 1, 1, 1, 1, 1), True),      # AsShotNeutral
        (50717, 4, 1, (int(white),), True),           # WhiteLevel
        (50714, 3, 1, (0,), True),                    # BlackLevel
    ]
    tifffile.imwrite(path, raw, photometric=32803, planarconfig="contig",
                     extratags=extratags, compression=None)
    return path
