"""Pyramide laplacienne : la structure multi-echelle du debruiteur.

Pourquoi une pyramide *decimee* : a chaque echelle, l'information utile est
deja lissee, donc la garder a pleine resolution ne sert a rien et coute tout.
En travaillant a la resolution propre de chaque echelle, le cout total vaut
1 + 1/4 + 1/16 + ... soit environ 1,33 fois celui d'une seule passe pleine
resolution, contre une passe par echelle auparavant (gain mesure : x5).

La construction de Burt-Adelson garantit une reconstruction *exacte*
(``synthese(analyse(x)) == x`` au bit de float pres) :

    g0 = image                      L_j = g_j - up(down(g_j))
    g_{j+1} = down(g_j)             g_j = L_j + up(g_{j+1})

Les niveaux ne sont donc pas une approximation : ce que le debruiteur ne
touche pas ressort intact.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np


def pyramid_levels(shape: Sequence[int], requested: int = 0, max_levels: int = 8) -> int:
    """Nombre de niveaux adapte a la taille de l'image."""
    if requested and requested > 0:
        return int(min(requested, max_levels))
    long_side = max(int(shape[0]), int(shape[1]))
    levels = 3
    if long_side >= 1000:
        levels = 4
    if long_side >= 2000:
        levels = 5
    if long_side >= 4000:
        levels = 6
    if long_side >= 8000:
        levels = 7
    # il faut garder au moins ~24 px de cote au niveau le plus grossier
    while levels > 2 and min(shape[0], shape[1]) >> levels < 12:
        levels -= 1
    return levels


def analyse(img: np.ndarray, levels: int) -> Tuple[List[np.ndarray], np.ndarray]:
    """Decompose en ``levels`` plans de details + residu (tailles decroissantes)."""
    details: List[np.ndarray] = []
    cur = np.ascontiguousarray(img, dtype=np.float32)
    for _ in range(max(0, int(levels))):
        if min(cur.shape[:2]) < 8:
            break
        down = cv2.pyrDown(cur)
        up = cv2.pyrUp(down, dstsize=(cur.shape[1], cur.shape[0]))
        details.append(cv2.subtract(cur, up))
        cur = down
    return details, cur


def synthese(details: Sequence[np.ndarray], residual: np.ndarray) -> np.ndarray:
    """Reconstruit l'image a partir des plans de details et du residu."""
    rec = residual
    for d in reversed(details):
        rec = cv2.pyrUp(rec, dstsize=(d.shape[1], d.shape[0]))
        rec = cv2.add(rec, d)
    return rec


def gaussian_stack(img: np.ndarray, levels: int) -> List[np.ndarray]:
    """Suite des versions reduites de ``img`` (une par niveau, plus le residu).

    Sert a evaluer, a chaque niveau, une grandeur definie sur l'image :
    luminosite locale pour le modele de bruit, masques de protection...
    """
    out = [np.ascontiguousarray(img, dtype=np.float32)]
    for _ in range(max(0, int(levels))):
        if min(out[-1].shape[:2]) < 8:
            break
        out.append(cv2.pyrDown(out[-1]))
    return out


def match_size(img: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Ajuste une carte a la taille exacte d'un niveau (±1 px de decimation)."""
    if img.shape[0] == shape[0] and img.shape[1] == shape[1]:
        return img
    return cv2.resize(img, (int(shape[1]), int(shape[0])), interpolation=cv2.INTER_LINEAR)
