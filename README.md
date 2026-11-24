# naturaw

Récupération de fichiers RAW abîmés : débruitage **naturel**, pixels chauds,
netteté de capture, ouverture des ombres, déflou de bougé et redressement des
photos penchées — en ligne de commande ou avec une petite interface graphique
locale.

Le mot d'ordre est *naturel* : une photo traitée ne doit pas se voir. Pas de
rendu lissé « plastique », pas de contours cernés, pas de couleurs qui bavent.
Chaque étape mesure ce qu'elle a devant elle et refuse d'agir quand elle n'est
pas sûre, plutôt que d'abîmer l'image.

Sur une photo sous-exposée de 2,5 diaphragmes, prise à très haut ISO et avec
des pixels chauds, la chaîne complète fait passer la ressemblance à la scène
réelle (SSIM) de **0,39 à 0,90**, en **0,55 s** pour 1,8 Mpx. Une photo de
24 Mpx est traitée en **5,5 s** sur quatre cœurs (3,6 s en mode rapide).

```
naturaw photo.CR2                    # une photo
naturaw dossier/ -o sorties/ -j 4    # tout un dossier, 4 en parallèle
naturaw photo.CR2 --rapide           # priorité à la vitesse
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

La chaîne enchaîne sept étapes, dans cet ordre imposé par la physique :
pixels chauds → bandes → exposition → débruitage → déflou → netteté →
ouverture des ombres → redressement. Chacune peut être coupée.

### 1. Défauts du capteur

**Pixels chauds et bruit impulsionnel** (poses longues, forts ISO, capteur
chaud). Ce sont des points isolés que le débruiteur prend pour du détail et
que la netteté transforme en petites étoiles. Un pixel n'est corrigé que s'il
sort du *maximum et du minimum* de ses huit voisins, pas d'une moyenne : un
détail fin, même très contrasté, est porté par plusieurs pixels et survit.
Mesure : 2 160 pixels chauds corrigés, +21 dB sur l'image concernée, et une
image saine ressort à 120 dB (c'est-à-dire inchangée). Un échantillonnage
préalable évite la passe complète quand il n'y a rien à corriger.

**Bandes de lecture** du capteur (rayures horizontales ou verticales) :
mesurées ligne par ligne sur les seuls pixels plats, rétrécies par la
corrélation entre les deux moitiés de chaque ligne. Désactivées par défaut
(voir « Limites »).

Une correction qui vaut d'être signalée : par défaut LibRaw abaisse le niveau
de blanc jusqu'au pixel le plus clair du fichier. Deux photos de la même scène
sortent alors avec des expositions différentes selon le bruit ou un reflet
(mesuré : 0,279 contre 0,360 de luminance moyenne, soit 29 % d'écart).
naturaw garde le niveau de blanc déclaré et décide lui-même de l'exposition.

### 2. Débruitage

Le bruit d'un capteur dépend du signal (`variance = a·I + b` : photons +
électronique) et le dématriçage le rend corrélé dans l'espace. Un seuil unique
sur toute l'image sur-lisse les ombres ou laisse du bruit dans les clairs.
naturaw **mesure** donc le bruit réellement présent, canal par canal, sur les
blocs les plus plats de l'image. Aux niveaux grossiers de la pyramide, la
mesure n'est plus possible : sur une image riche en texture, *aucun* bloc n'y
est vide de structure et l'estimation monte jusqu'à vingt fois trop haut — ce
qui efface le détail. Le bruit y est donc **prédit** à partir du niveau fin,
avec une loi de décroissance mesurée sur des captures simulées (×0,08 au
premier passage, ×0,25 ensuite) remarquablement stable d'une scène et d'une
sensibilité à l'autre ; la mesure ne sert plus que de garde-fou quand elle est
*plus basse* que la prédiction. Ce seul changement a rendu 1 dB et 4 points de
MTF de texture sur la mire dead leaves.

Enfin, la force est **ramenée à zéro quand le bruit mesuré est négligeable** :
à 200 ISO il n'y a rien à gagner et du détail à perdre, et la photo ressort
bit pour bit identique.

Le reste applique à chaque niveau un **filtre de Wiener local**, en
**deux passes** :

```
passe 1 :  gain = max(0, (E − var_bruit) / E)            E = énergie locale
passe 2 :  gain = E_estimée / (E_estimée + var_bruit)    E_estimée = passe 1
```

La seconde passe est l'idée du second étage de BM3D : la première sait *où*
est le signal, la seconde peut donc doser beaucoup plus finement. Gain mesuré
à très haut ISO : SSIM 0,858 → 0,899, et des aplats deux fois plus propres,
pour un coût négligeable.

Cinq choix font le rendu naturel :

* **Luminance et chrominance séparées** (YCoCg). Le bruit de couleur est le
  plus visible et le moins porteur d'information : il est traité une fois et
  demie plus fort. La luminance, elle, est à peine touchée — c'est elle qui
  porte la netteté perçue.
* **Pyramide laplacienne** (Burt-Adelson) : reconstruction *exacte* au bit
  près, donc ce que le débruiteur ne touche pas ressort intact, et chaque
  niveau travaille à sa propre résolution — cinq fois moins de calcul qu'une
  décomposition pleine résolution, à qualité identique (mesuré).
* **Protection des textures** : un masque de contraste local rapporté au bruit
  local réinjecte une partie des coefficients dans les feuillages, tissus,
  cheveux, gravier — là où un débruiteur classique « efface ».
* **Grain conservé, et dosé par le bruit mesuré** : 18 % du grain fin de
  luminance est remis en place (jamais sur la couleur). Une image parfaitement
  lisse se *voit* ; un grain fin achromatique est ce qui rend le résultat
  crédible. Mais garder 18 % du bruit d'un 25600 ISO ne ferait que remettre du
  bruit : la part conservée décroît automatiquement avec le bruit mesuré
  (0,18 à bas ISO, 0,07 à σ = 5 %).

Un dernier garde-fou borne l'écart de chaque canal à cinq écarts-types de
bruit : un débruiteur multi-échelle peut sinon, au bord d'un aplat très
saturé, reconstituer une couleur franche à partir d'une échelle grossière.

### 3. Déflou de bougé

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

### 4. Netteté de capture

Un RAW est *toujours* un peu doux : filtre passe-bas devant le capteur,
dématriçage, diffraction, et le débruitage lui-même. Contrairement au bougé,
cette perte est faible, isotrope et prévisible : on peut la corriger sans
rien deviner. naturaw déconvolue une petite gaussienne par **filtre inverse
régularisé précalculé** — une seule convolution, 45 fois plus rapide que huit
itérations de Richardson-Lucy et plus précise (47,2 dB contre 43,4 sur un cas
de référence).

La correction est pondérée par la structure locale : là où il ne reste que du
bruit, rien n'est accentué. C'est ce qui la sépare d'un masque flou, qui
remonte le bruit autant que le détail (mesure : 36,6 dB / SSIM 0,963 contre
35,5 dB / 0,945).

### 5. Ouverture des ombres

C'est ce qui sauve vraiment une photo sous-exposée. Un gain global brûle le
ciel ; une courbe en S écrase le contraste local. naturaw sépare la luminance
en une **base** (l'éclairement perçu) et un **détail** (la matière), dans le
domaine logarithmique où le contraste est multiplicatif, et ne modifie que la
base. La base vient d'un filtre guidé, qui suit les contours : c'est ce qui
évite le halo clair autour des objets sombres, défaut classique de ces outils
(mesuré : variation de gain < 5 % le long d'un contour franc). Le gain est
appliqué en rapport sur les trois canaux, donc à teinte constante.

À −3 EV, la médiane passe de 0,025 à 0,131 sans qu'aucune haute lumière ne
soit écrêtée.

### 6. Redressement

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

Les mesures sont faites sur des **captures simulées** : une scène de synthèse
est échantillonnée par une matrice de Bayer, le bruit de photons et de lecture
est ajouté *au niveau du photosite*, le résultat est écrit en DNG puis
redéveloppé par LibRaw comme n'importe quel fichier d'appareil. Le bruit y est
donc corrélé spatialement et coloré par le dématriçage (corrélation mesurée
entre pixels voisins : 0,3 — un bruit blanc ajouté après coup donnerait 0,0 et
serait beaucoup plus facile à enlever). La référence est la même scène
développée sans bruit. `pytest` rejoue ces mesures.

Deux scènes : une scène d'architecture (ciel, façades, aplats saturés) et une
mire **« dead leaves »** — des disques empilés dont les rayons suivent une loi
de puissance. C'est le modèle de référence des statistiques d'images
naturelles et la mire que l'industrie photo utilise pour mesurer la
conservation de la texture (spectre en 1/f², contours francs à toutes les
échelles). C'est le cas difficile : aucune zone n'y est lisse.

**Scène d'architecture** (PSNR / SSIM) :

| sensibilité | fichier développé | naturaw | référence `cv2` NLM |
|---|---|---|---|
| 200 ISO | 48,6 dB / 0,992 | 48,6 dB / 0,992 *(laissée intacte)* | 41,3 dB / 0,981 |
| 1600 ISO | 39,7 dB / 0,940 | **45,0 dB / 0,985** | 41,2 dB / 0,981 |
| 6400 ISO | 33,2 dB / 0,786 | **40,2 dB / 0,963** | 38,1 dB / 0,969 |
| 25600 ISO | 25,9 dB / 0,452 | **34,6 dB / 0,893** | 33,9 dB / 0,946 |
| 102400 ISO | 17,8 dB / 0,148 | **28,4 dB / 0,740** | 26,6 dB / 0,869 |

**Mire dead leaves** — le cas où la plupart des débruiteurs s'effondrent :

| sensibilité | fichier développé | naturaw | référence `cv2` NLM |
|---|---|---|---|
| 200 ISO | 32,1 dB / 0,991 | 31,5 dB / 0,988 | 31,4 dB / 0,977 |
| 1600 ISO | 31,0 dB / 0,974 | 30,4 dB / **0,981** | 30,6 dB / 0,971 |
| 6400 ISO | 28,2 dB / 0,911 | **28,5 dB / 0,963** | 27,7 dB / 0,917 |
| 25600 ISO | 21,8 dB / 0,731 | **24,4 dB / 0,900** | 21,3 dB / 0,714 |
| 102400 ISO | 16,8 dB / 0,536 | **19,3 dB / 0,740** | 17,5 dB / 0,560 |

**Conservation de la texture (MTF, méthode du spectre croisé)** : combien du
*signal* d'origine subsiste à chaque finesse de détail. Le spectre croisé
ignore le bruit résiduel, donc on ne peut pas tricher en gardant du bruit à la
place de la texture. 1,00 = texture intacte.

| | fin (0,35 cy/px) | moyen (0,20) | large (0,05) |
|---|---|---|---|
| naturaw, 6400 ISO | 0,94 | 0,95 | 0,98 |
| naturaw, 25600 ISO | 0,87 | 0,90 | 0,95 |
| naturaw, 102400 ISO | 0,62 | 0,68 | 0,81 |
| NLM, 102400 ISO (architecture) | 0,21 | 0,28 | 0,83 |

**Sauvetage complet** — photo sous-exposée de 2,5 EV, bruit σ = 5,4 %, pixels
chauds, douceur de capteur, penchée de 2° :

| | PSNR | SSIM |
|---|---|---|
| développement brut (gain global seul) | 21,6 dB | 0,394 |
| **naturaw, réglages par défaut** | **29,4 dB** | **0,904** |

Le rapport de traitement dit exactement ce qui a été fait : 2 160 pixels
chauds corrigés, bruit mesuré à 5,36 %, grain ramené de 0,18 à 0,067,
exposition +2,50 EV, bougé non détecté, netteté 0,60, redressement +2,00°
(valeur réelle : 2,00°).

**Netteté de capture** — scène passée dans une douceur de capteur
(gaussienne σ = 0,75) puis bruitée, après débruitage : 36,0 → **39,2 dB** à
bas bruit, 34,7 → **36,2 dB** à 2 % de bruit, contre 35,5 dB / SSIM 0,945 pour
un masque flou classique.

**Déflou de bougé**, sur captures simulées (le flou est appliqué *avant* le
capteur, comme dans la réalité) et après débruitage :

| bougé | 1600 ISO | 6400 ISO | 25600 ISO |
|---|---|---|---|
| rectiligne 9 px | +1,18 dB / SSIM +0,125 | +1,07 / +0,115 | +0,40 / +0,050 |
| rectiligne 15 px | +1,06 / +0,123 | +0,96 / +0,116 | +0,54 / +0,066 |
| bougé de main 25 px | +0,18 / +0,023 | +0,05 / +0,007 | *non détecté* |

Six images nettes sur six sont laissées **strictement inchangées**, de même
qu'un flou de mise au point ou une estimation instable.

Ces chiffres valent plusieurs fois ceux de la version précédente, grâce à une
décision : le noyau estimé est remplacé par la **trajectoire rectiligne
de même étendue et même orientation**. Un noyau libre estimé sur quelques
imagettes bruitées porte des lobes parasites qui font diverger la
déconvolution ; une droite propre récupère jusqu'à *la totalité* du gain
qu'aurait donné le vrai noyau (mesuré : +1,06 dB contre +1,06 dB avec le
noyau exact sur un bougé de 15 px). Pour un bougé en courbe, le noyau libre
reste disponible : `--deflou-noyau-libre`.

**La dose suit la confiance.** Les risques ne sont pas symétriques : une
déconvolution juste rapporte de 0,3 à 1,8 dB, une déconvolution fondée sur une
estimation fausse en coûte 6,8 (mesuré sur une scène où l'estimateur se fait
piéger par des rangées de fenêtres régulières — la périodicité est le piège
classique de la déconvolution aveugle). La dose est donc proportionnelle à
l'anisotropie du noyau trouvé : pleine au-delà de 2,6, nulle en dessous de
1,8. Les cas francs gardent tout, les cas douteux ne reçoivent presque rien,
et le cas piégé ressort **bit pour bit identique** à l'original.

Cinq autres façons de distinguer une vraie trainée d'un faux positif ont été
essayées et mesurées sans succès sur ce type de scène (ajustement du
log-spectre, cohérence d'échelle, contraste directionnel normalisé,
auto-cohérence après déconvolution d'essai, périodicité directionnelle).
C'est pourquoi la prudence l'emporte ici sur l'ambition.

**Redressement** : erreur inférieure à 0,05° entre 0,3° et 7,5° d'inclinaison ;
une scène sans structure n'est pas touchée.

### Vitesse

Quatre cœurs, chaîne complète, du RAW décodé au fichier écrit :

| image | par défaut | `--rapide` |
|---|---|---|
| 6 Mpx | 2,6 s | 1,0 s |
| 12 Mpx | 2,7 s | 1,8 s |
| 24 Mpx | 5,5 s | 3,6 s |
| 50 Mpx | 8,8 s | 8,8 s |

Soit **0,23 s/Mpx** par défaut. Ce qui a été fait pour y arriver, chiffres à
l'appui sur 24 Mpx :

* pyramide laplacienne décimée au lieu d'une décomposition pleine résolution
  à chaque échelle : **×5** sur la décomposition ;
* cartes de gain et de structure calculées en demi-résolution (elles sont
  lisses par construction) : débruitage 6,7 s → 3,8 s ;
* les trois canaux traités de front (OpenCV et numpy relâchent le GIL) :
  1,17 s → 0,53 s ;
* conversions couleur par `cv2.transform` plutôt que canal par canal :
  0,88 s → 0,15 s ;
* mesure du bruit sur un échantillon de blocs : 4,7 s → 0,6 s ;
* filtre inverse précalculé au lieu de Richardson-Lucy pour la netteté :
  **×45** ;
* pré-tests avant les étapes coûteuses : les pixels chauds ne sont cherchés
  partout que si un échantillon en contient (1,4 s → 0,06 s), et l'analyse de
  bougé s'arrête sur une seule imagette quand l'image est manifestement nette.

Mémoire : environ huit fois la taille de l'image en float32 (≈ 2,5 Go pour un
60 Mpx). `--cote-max` ou `--demi-taille` pour les machines justes.

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
  distinguent pas du plancher de l'estimateur. Et sur une scène pauvre en
  texture, il refuse souvent d'agir même quand il y a du bougé (7 cas sur 9
  dans le banc d'essai sur scène d'architecture) : il préfère ne rien faire
  plutôt que d'appliquer un noyau faux.
* La **mise au point ratée** n'est pas corrigée (flou isotrope) : c'est un
  autre problème, et le déconvoluer à l'aveugle abîme plus qu'il ne répare.
* Le **traitement par lot en parallèle** (`-j`) démarre ses processus en mode
  `forkserver` ou `spawn`. Si vous appelez `process_many` depuis votre propre
  script Python, entourez l'appel d'un `if __name__ == "__main__":` — c'est la
  règle habituelle de `multiprocessing`, et sans elle le script se relance
  dans chaque processus de travail.
* La **mesure du bruit est calibrée pour des fichiers d'appareil** (12 Mpx et
  plus). Sur une image fortement réduite et très texturée, elle surestime le
  bruit (×1,7 mesuré sur une mire dead leaves de 0,5 Mpx) et le débruitage
  devient trop appuyé ; `-d leger` compense.
* La **correction des bandes** est désactivée par défaut. Sur une scène dont
  les structures sont parfaitement alignées sur la grille du capteur (façade,
  mire), elle confond contenu et rayures ; sur du contenu ordinaire elle
  gagne environ 2,4 dB. À activer quand on *voit* des rayures
  (`--bandes h`, `v` ou `hv`).
* La **netteté de capture** suppose un fichier venant d'un vrai capteur, donc
  toujours un peu doux. Sur une image déjà parfaitement nette, elle
  sur-accentue : `--nettete 0` la coupe.
* L'**ouverture automatique des ombres** suit l'histogramme ; sur une photo
  volontairement sombre (nocturne, clair-obscur), `--ombres non`.
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

L'aperçu donne **deux vues** : l'image entière réduite (cadrage, exposition,
redressement) et une **zone à 100 %** prise dans la partie la plus texturée,
traitée exactement comme le fichier final. C'est la seule vue sur laquelle on
puisse juger le bruit et la netteté — une image réduite cache les deux. En
ligne de commande, `--apercu` écrit les deux fichiers (`_apercu.jpg` et
`_apercu-100.jpg`).

### Ligne de commande

```bash
naturaw photo.NEF -d fort --grain 0.25          # débruitage appuyé, grain argentique
naturaw photo.CR3 -d sauvetage --ombres 0.8 --nettete 0.8   # photo « inexploitable »
naturaw photo.NEF --rapide                      # deux fois plus rapide, pour trier
naturaw photo.ARW --bandes h                    # rayures de lecture du capteur
naturaw photo.ARW --apercu                      # contrôle rapide (demi-résolution, JPEG)
naturaw photo.RAF --deflou-longueur 12 --deflou-angle 35
naturaw photo.DNG --angle -1.4 --recadrage max  # redressement imposé
naturaw dossier/ -r -o sorties/ -f jpeg -q 95 -j 4
naturaw photo.CR2 -d off -b off                 # redressement seul
```

Principales options (`naturaw --help` pour la liste complète) :

| option | rôle | défaut |
|---|---|---|
| `-d, --debruitage` | `off`, `leger`, `normal`, `fort`, `extreme`, `sauvetage` ou un nombre | `normal` |
| `--grain` | part du grain fin conservée (0 à 0,4) | `0.16` |
| `--detail` | protection des zones texturées | `0.45` |
| `--chroma` | force relative sur la couleur | `1.5` |
| `--nettete`, `--nettete-rayon` | netteté de capture (0 la coupe) | `0.6` / `0.8 px` |
| `--ombres` | `auto`, `non`, ou 0 à 1 (1 ≈ +3 EV dans les noirs) | `auto` |
| `--recup-hautes-lumieres`, `--contraste-local` | 0 à 1 | `0` |
| `--pixels-chauds` | `auto` ou `off` | `auto` |
| `--bandes` | `off`, `auto`, `h`, `v`, `hv` | `off` |
| `--rapide` | une passe de débruitage, pas d'analyse de bougé | — |
| `--apercu` | vue d'ensemble + zone à 100 %, en JPEG | — |
| `-b, --deflou` | `off`, `auto`, `on`, `force` | `auto` |
| `--deflou-dose` | dose de déconvolution | `0.8` |
| `--deflou-longueur`, `--deflou-angle` | bougé connu, imposé | — |
| `--deflou-noyau-libre` | noyau libre au lieu de la trajectoire rectiligne | — |
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
[1/1] essai.dng  (1650x1100, 1.8 Mpx)
   bruit mesure      : Y=5.36%  Co=5.96%  Cg=5.23%
   pixels chauds     : 2160 pixels corriges (0.0397 %)
   debruitage        : force 1.00, chroma x1.5, grain 0.18, 2 passes, 4 niveaux
   exposition        : +2.50 EV
   deflou            : ignore : image deja nette (etendue 1.3 px)
   nettete           : dose 0.60, rayon 0.8 px
   redressement      : redressee de +2.00 deg (confiance 0.97, 90 % conserve)
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
| `naturaw/pyramid.py` | pyramide laplacienne à reconstruction exacte |
| `naturaw/noise.py` | mesure du bruit par canal et par niveau |
| `naturaw/denoise.py` | Wiener local en deux passes, grain, protection des textures |
| `naturaw/rescue.py` | pixels chauds, bandes de lecture |
| `naturaw/sharpen.py` | netteté de capture (filtre inverse régularisé) |
| `naturaw/tone.py` | ouverture des ombres sans halo |
| `naturaw/deblur.py` | estimation aveugle du noyau, validation, Richardson-Lucy |
| `naturaw/deskew.py` | détection de l'inclinaison, rotation, recadrage |
| `naturaw/filters.py` | starlet, filtre guidé, bilatéral joint |
| `naturaw/colorspace.py` | YCoCg, courbes de transfert |
| `naturaw/rawio.py` | lecture LibRaw, écriture TIFF/PNG/JPEG |
| `naturaw/pipeline.py` | enchaînement complet et rapports |
| `naturaw/cli.py`, `naturaw/webui.py` | ligne de commande, interface locale |
| `tests/` | 98 tests, dont la fabrication de RAW de synthèse |

## Tests

```bash
pip install -e ".[dev]"
pytest -q            # 98 tests, environ 9 s
```

Les tests ne se contentent pas de vérifier que le code s'exécute : ils
mesurent le gain de PSNR/SSIM, la texture conservée, le respect du garde-fou
à 5 σ, la précision du redressement, la reconstruction exacte de la pyramide,
l'adjonction correcte des opérateurs de convolution, l'absence de halo autour
d'un contour franc, la survie d'un détail fin au filtre de pixels chauds, et
rejouent la chaîne complète sur un DNG Bayer fabriqué pour l'occasion.

## Licence

MIT.
