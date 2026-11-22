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
