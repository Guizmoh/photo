# naturaw

Débruitage **naturel** de fichiers RAW, déflou de bougé et redressement
automatique des photos penchées — en ligne de commande ou avec une petite
interface graphique locale.

Le mot d'ordre est *naturel* : une photo traitée ne doit pas se voir. Pas de
rendu lissé « plastique », pas de contours cernés, pas de couleurs qui bavent.
Chaque étape mesure ce qu'elle a devant elle et refuse d'agir quand elle n'est
pas sûre, plutôt que d'abîmer l'image.

```
naturaw photo.CR2                    # une photo
naturaw dossier/ -o sorties/ -j 4    # tout un dossier, 4 en parallèle
naturaw --interface                  # interface graphique dans le navigateur
```

---

## Installation

Python 3.9 ou plus récent.

```bash
git clone <ce dépôt> && cd photo
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -e .
```

Dépendances : `numpy`, `scipy`, `opencv-python-headless`, `rawpy` (LibRaw),
`tifffile`, `imageio`. L'interface graphique n'utilise que la bibliothèque
standard — rien à installer de plus, et rien ne sort de la machine.

Formats lus : tous les RAW gérés par LibRaw (CR2, CR3, NEF, ARW, RAF, ORF,
RW2, DNG, PEF, SRW, IIQ…) ainsi que TIFF/PNG/JPEG déjà développés.
Formats écrits : TIFF 16 bits (défaut), PNG 16 ou 8 bits, JPEG.

Pas de photo sous la main pour essayer ? Le dépôt sait fabriquer un vrai RAW
de test :

```bash
python exemples/creer_raw_test.py /tmp/essai.dng --bruit eleve --bouge 11 --penche -2.3
naturaw /tmp/essai.dng --espace raw -f jpeg
```

---

## Ce que fait le logiciel, et comment

Tout le traitement a lieu en **lumière linéaire**, juste après le dématriçage
et avant toute courbe de contraste : c'est le seul espace où le modèle de
bruit du capteur et la déconvolution sont physiquement justes.

### 1. Débruitage

Le bruit d'un capteur dépend du signal (`variance = a·I + b` : photons +
électronique) et le dématriçage le rend corrélé dans l'espace. Un seuil unique
sur toute l'image sur-lisse les ombres ou laisse du bruit dans les clairs.
naturaw **mesure** donc le bruit réellement présent, canal par canal et
échelle par échelle, sur les blocs les plus plats de l'image, puis applique à
chaque échelle un **filtre de Wiener local** :

```
gain = max(0, (énergie_locale − variance_du_bruit) / énergie_locale)
```

Quatre choix font le rendu naturel :

* **Luminance et chrominance séparées** (YCoCg). Le bruit de couleur est le
  plus visible et le moins porteur d'information : il est traité une fois et
  demie plus fort. La luminance, elle, est à peine touchée — c'est elle qui
  porte la netteté perçue.
* **Transformée « starlet » non décimée** (algorithme à trous, noyau
  B3-spline) plutôt que des ondelettes classiques : isotrope, invariante par
  translation, reconstruction par simple somme. Aucun artefact en damier,
  aucun halo de contour.
* **Protection des textures** : un masque de contraste local rapporté au bruit
  local réinjecte une partie des coefficients dans les feuillages, tissus,
  cheveux, gravier — là où un débruiteur classique « efface ».
* **Grain conservé** : 16 % du grain fin de luminance est remis en place
  (jamais sur la couleur). Une image parfaitement lisse se *voit* ; un grain
  fin achromatique est ce qui rend le résultat crédible. Réglable de 0 à 0,4.

Un dernier garde-fou borne l'écart de chaque canal à cinq écarts-types de
bruit : un débruiteur multi-échelle peut sinon, au bord d'un aplat très
saturé, reconstituer une couleur franche à partir d'une échelle grossière.

### 2. Déflou de bougé

Le bougé est modélisé par une convolution avec la trajectoire du capteur
pendant la pose, inconnue : c'est un problème *aveugle*.

* **Estimation du noyau** par minimisation alternée dans le domaine des
  gradients, du grossier vers le fin (approche Cho & Lee / Xu & Jia
  simplifiée), sur plusieurs imagettes **fenêtrées** prises dans les zones à
  la fois contrastées et riches en orientations. Le fenêtrage de Hann est
  déterminant : sans lui, le repliement circulaire de la FFT fausse tout
  (corrélation du noyau 0,89 au lieu de 1,00 sur un cas de référence). Les
  imagettes sont traitées **conjointement**, car le noyau est commun à l'image
  entière alors qu'une zone isolée (rien que des contours horizontaux) ne le
  contraint que partiellement.
* **Validation** avant d'agir, en mode `auto` : le flou doit être franc
  (au-dessus du plancher de l'estimateur, ~1,5 px), nettement **directionnel**
  (un flou isotrope est une erreur de mise au point, que l'on ne cherche pas à
  corriger) et **stable** (deux groupes d'imagettes disjoints doivent trouver
  le même noyau). Sinon : rien n'est appliqué, et la raison est affichée.
* **Déconvolution** par Richardson-Lucy (adapté au bruit de photons) régularisé
  par variation totale, à dose 0,8 par défaut, avec deux anti-halos : la
  correction est bornée par l'intervalle local de l'image d'origine, et elle
  est appliquée en *rapport de luminance*, ce qui laisse teinte et saturation
  exactement intactes (pas de frange colorée).

### 3. Redressement

Les lignes dominantes d'une scène construite sont horizontales ou verticales.
Les segments sont détectés (LSD, repli sur Hough), ramenés **modulo 90°** — une
verticale penchée de 2° et une horizontale penchée de 2° disent la même chose —
et l'histogramme pondéré par le carré de la longueur donne l'angle. Rotation
Lanczos, puis recadrage dans le plus grand cadre **de même format** inscrit
dans l'image tournée (ou à l'aire maximale, au choix). Un indice de confiance
accompagne la mesure : sous le seuil, l'image n'est pas touchée.

L'ordre des opérations est imposé par la physique : débruiter **avant** de
déconvoluer (sinon on amplifie le bruit), déconvoluer **avant** de tourner (la
convolution n'est valable que dans la géométrie d'origine), et une seule
interpolation géométrique en tout et pour tout.

---

## Résultats mesurés

Sur scènes de synthèse avec bruit réaliste dépendant du signal (métriques
calculées en sRGB, vérité terrain connue ; `pytest` rejoue ces mesures).

**Débruitage** — scène d'architecture (ciel, façades, texture, aplats saturés) :

| bruit (σ à 18 % de gris) | image bruitée | naturaw | référence `cv2` NLM |
|---|---|---|---|
| 0,96 % | 38,8 dB / SSIM 0,963 | **45,8 dB / 0,991** | 37,3 dB / 0,967 |
| 2,06 % | 32,0 dB / SSIM 0,849 | **40,5 dB / 0,969** | 34,3 dB / 0,951 |
| 3,46 % | 27,2 dB / SSIM 0,668 | **37,2 dB / 0,931** | 31,1 dB / 0,930 |

Sur une scène contenant une mire de résolution et une texture fine, naturaw
conserve **92 à 108 % de l'énergie de détail** de l'image d'origine, contre
10 à 21 % pour le débruiteur non-local de référence : c'est toute la
différence entre une image propre et une image repassée.

**Déflou** (scène texturée, mode `auto`) : bougé rectiligne de 9 px, +1,7 dB et
SSIM 0,843 → 0,887 ; bougé de main de 25 px, +2,0 dB et SSIM 0,696 → 0,793.
Une image nette, un flou de mise au point ou une estimation instable sont
laissés **strictement inchangés**.

**Redressement** : erreur inférieure à 0,05° entre 0,3° et 7,5° d'inclinaison
sur la scène de test ; une scène sans structure n'est pas touchée.

**Vitesse** (4 cœurs) : ≈ 0,5 s/Mpx pour le débruitage, ≈ 1 s de plus si le
déflou s'applique. Environ 15 s pour un fichier 24 Mpx, mémoire ≈ 8 fois la
taille de l'image en float32.

---

## Limites, dites franchement

* **L'estimation aveugle du bougé n'est pas fiable sur toutes les scènes.**
  Elle marche bien sur des images texturées ; elle décroche sur les scènes
  pauvres en texture ou très répétitives (façade de fenêtres alignées, grand
  ciel uni). C'est une propriété du problème, pas un réglage à trouver. Le
  mode `auto` préfère alors ne rien faire et le dire. Si vous *voyez* la
  traînée, donnez-la : `--deflou-longueur 12 --deflou-angle 35` est la voie la
  plus sûre et la plus efficace. `-b on` assouplit les tests, `-b force` les
  supprime (au risque de dégrader une image nette).
* Le déflou **auto** ne détecte pas les bougés inférieurs à ~6 px : ils ne se
  distinguent pas du plancher de l'estimateur.
* La **mise au point ratée** n'est pas corrigée (flou isotrope) : c'est un
  autre problème, et le déconvoluer à l'aveugle abîme plus qu'il ne répare.
* Aucun **profil ICC** n'est incorporé aux fichiers écrits. La sortie par
  défaut est en sRGB ; les autres espaces (`--espace adobe|prophoto|raw`) sont
  pour qui gère sa couleur lui-même.
* Les **métadonnées EXIF ne sont pas recopiées** dans le fichier de sortie ;
  un résumé du traitement est écrit dans la description du TIFF, et
  `--rapport` produit un JSON complet.
* La validation s'appuie sur des RAW de **synthèse** (DNG Bayer fabriqués par
  le dépôt), faute de fichiers d'appareil dans l'environnement de
  développement. Les réglages par défaut sont donc calés sur un bruit
  gaussien dépendant du signal ; sur vos fichiers, commencez par `--apercu`.

---

## Utilisation

### Interface graphique

```bash
naturaw --interface          # puis http://127.0.0.1:8731/
```

Parcours des dossiers, sélection des photos, réglages avec curseurs, **aperçu
avant/après** avec comparateur glissant, traitement par lot avec avancement.
Le serveur n'écoute que sur la machine locale.

### Ligne de commande

```bash
naturaw photo.NEF -d fort --grain 0.25          # débruitage appuyé, grain argentique
naturaw photo.ARW --apercu                      # contrôle rapide (demi-résolution, JPEG)
naturaw photo.RAF --deflou-longueur 12 --deflou-angle 35
naturaw photo.DNG --angle -1.4 --recadrage max  # redressement imposé
naturaw dossier/ -r -o sorties/ -f jpeg -q 95 -j 4
naturaw photo.CR2 -d off -b off                 # redressement seul
```

Principales options (`naturaw --help` pour la liste complète) :

| option | rôle | défaut |
|---|---|---|
| `-d, --debruitage` | `off`, `leger`, `normal`, `fort`, `extreme` ou un nombre | `normal` |
| `--grain` | part du grain fin conservée (0 à 0,4) | `0.16` |
| `--detail` | protection des zones texturées | `0.45` |
| `--chroma` | force relative sur la couleur | `1.5` |
| `-b, --deflou` | `off`, `auto`, `on`, `force` | `auto` |
| `--deflou-dose` | dose de déconvolution | `0.8` |
| `--deflou-longueur`, `--deflou-angle` | bougé connu, imposé | — |
| `-s, --redressement` | `auto` ou `off` | `auto` |
| `--angle`, `--angle-max`, `--confiance` | redressement imposé / limites | — / `8°` / `0.22` |
| `--recadrage` | `aspect`, `max`, `none` | `aspect` |
| `-f, --format` | `tiff16`, `tiff8`, `png16`, `png`, `jpeg` | `tiff16` |
| `--exposition` | `auto`, `non`, ou une correction en EV | `auto` |
| `--cote-max`, `--demi-taille` | traitement allégé | — |
| `-j, --jobs` | fichiers traités en parallèle | `1` |
| `--rapport` | rapport JSON détaillé | — |

Chaque fichier traité affiche ce qui a été mesuré et décidé :

```
[1/1] essai.dng  (1800x1200, 2.2 Mpx)
   bruit mesure      : Y=1.54%  Co=0.66%  Cg=0.29%
   exposition        : +0.16 EV
   deflou            : ignore : estimation instable sur cette scene (...)
   redressement      : redressee de +2.31 deg (confiance 0.99, 89 % conserve)
```

### Comme bibliothèque

```python
from naturaw import Settings, DenoiseParams, process_file, denoise_linear_rgb

reglages = Settings(denoise=DenoiseParams(strength=1.2, grain=0.2))
rapport = process_file("photo.CR2", "photo.tif", reglages)
print(rapport["noise_sigma"], rapport["deskew"])

# ou directement sur un tableau RGB linéaire float32
propre = denoise_linear_rgb(image_lineaire, DenoiseParams())
```

---

## Organisation du code

| fichier | rôle |
|---|---|
| `naturaw/noise.py` | mesure du bruit par canal et par échelle |
| `naturaw/denoise.py` | Wiener local multi-échelle, grain, protection des textures |
| `naturaw/deblur.py` | estimation aveugle du noyau, validation, Richardson-Lucy |
| `naturaw/deskew.py` | détection de l'inclinaison, rotation, recadrage |
| `naturaw/filters.py` | starlet, filtre guidé, bilatéral joint |
| `naturaw/colorspace.py` | YCoCg, courbes de transfert |
| `naturaw/rawio.py` | lecture LibRaw, écriture TIFF/PNG/JPEG |
| `naturaw/pipeline.py` | enchaînement complet et rapports |
| `naturaw/cli.py`, `naturaw/webui.py` | ligne de commande, interface locale |
| `tests/` | 70 tests, dont la fabrication de RAW de synthèse |

## Tests

```bash
pip install -e ".[dev]"
pytest -q            # 70 tests, environ 8 s
```

Les tests ne se contentent pas de vérifier que le code s'exécute : ils
mesurent le gain de PSNR/SSIM, la texture conservée, le respect du garde-fou
à 5 σ, la précision du redressement, l'adjonction correcte des opérateurs de
convolution, et rejouent la chaîne complète sur un DNG Bayer fabriqué pour
l'occasion.

## Licence

MIT.
