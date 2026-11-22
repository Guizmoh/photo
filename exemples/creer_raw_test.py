#!/usr/bin/env python3
"""Fabrique un RAW de test (DNG) degrade, pour essayer naturaw sans photo perso.

    python exemples/creer_raw_test.py /tmp/essai.dng --bruit eleve --bouge 11 --penche -2.3
    naturaw /tmp/essai.dng --espace raw -f jpeg

Le fichier produit est un vrai DNG Bayer, lu par LibRaw comme n'importe quel
RAW d'appareil.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import synth  # noqa: E402
from naturaw.deblur import convolve, linear_motion_kernel  # noqa: E402
from naturaw.deskew import rotate_image  # noqa: E402

NIVEAUX = {"aucun": (0.0, 0.0), "faible": (2e-4, 1e-5),
           "moyen": (8e-4, 4e-5), "eleve": (1.8e-3, 1e-4), "extreme": (5e-3, 3e-4)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sortie", help="chemin du DNG a creer")
    p.add_argument("--largeur", type=int, default=1800)
    p.add_argument("--hauteur", type=int, default=1200)
    p.add_argument("--bruit", default="eleve", choices=list(NIVEAUX))
    p.add_argument("--bouge", type=float, default=0.0, metavar="PX",
                   help="longueur du flou de bouge en pixels (0 = net)")
    p.add_argument("--bouge-angle", type=float, default=25.0)
    p.add_argument("--penche", type=float, default=-2.3, metavar="DEG",
                   help="inclinaison appliquee a la scene")
    p.add_argument("--reference", default="", metavar="NPY",
                   help="enregistre aussi la scene nette (.npy) pour comparer")
    a = p.parse_args()

    scene = synth.architecture(a.hauteur, a.largeur)
    if a.reference:
        np.save(a.reference, scene)
    img = rotate_image(scene, a.penche, crop="none") if abs(a.penche) > 1e-3 else scene
    if a.bouge > 0:
        k = linear_motion_kernel(a.bouge, a.bouge_angle)
        img = np.stack([convolve(img[..., c], k) for c in range(3)], -1)
    noise_a, noise_b = NIVEAUX[a.bruit]
    if noise_a > 0:
        img = synth.add_photon_noise(img, noise_a, noise_b, seed=11)
    synth.write_dng(a.sortie, img)
    print(f"Ecrit : {a.sortie}  ({a.largeur}x{a.hauteur}, bruit {a.bruit}, "
          f"bouge {a.bouge:g} px, inclinaison {a.penche:+g} deg)")
    print("Essai :  naturaw", a.sortie, "--espace raw -f jpeg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
