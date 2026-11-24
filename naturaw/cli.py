"""Interface en ligne de commande de naturaw."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from . import __version__
from .deblur import DeblurParams
from .denoise import PRESETS, DenoiseParams
from .deskew import DeskewParams
from .pipeline import Settings, collect_inputs, process_many, summarize
from .rescue import RescueParams
from .sharpen import SharpenParams
from .tone import ToneParams

EPILOGUE = """
exemples
--------
  naturaw photo.CR2
      debruitage naturel + deflou si un bouge est detecte + redressement,
      sortie TIFF 16 bits a cote de l'original.

  naturaw dossier/ -o sorties/ -f jpeg -q 95 -j 4
      tout un dossier en JPEG qualite 95, 4 fichiers a la fois.

  naturaw photo.NEF -d fort --grain 0.25
      debruitage appuye mais grain conserve (rendu argentique).

  naturaw photo.CR3 -d sauvetage --ombres 0.8 --nettete 0.8
      photo tres sous-exposee et tres bruitee : on sort tout.

  naturaw photo.NEF --rapide
      une passe de debruitage, pas d'analyse de bouge : environ deux fois
      plus rapide, pour trier un reportage.

  naturaw photo.ARW --bandes h
      rayures horizontales de lecture du capteur (pose longue, haut ISO).

  naturaw photo.ARW -b on
      force le deflou meme si le bouge detecte est faible.

  naturaw photo.RAF --deflou-longueur 12 --deflou-angle 35
      bouge connu : trainee de 12 px orientee a 35 degres.

  naturaw photo.DNG --angle -1.4 --recadrage max
      redressement impose de 1.4 degre, recadrage a l'aire maximale.

  naturaw --interface
      ouvre l'interface graphique dans le navigateur.
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="naturaw",
        description="Debruitage naturel, deflou de bouge et redressement "
                    "automatique de fichiers RAW.",
        epilog=EPILOGUE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("entrees", nargs="*", metavar="ENTREE",
                   help="fichiers RAW (ou images) et/ou dossiers a traiter")

    g = p.add_argument_group("entrees / sorties")
    g.add_argument("-o", "--sortie", metavar="DOSSIER", default="",
                   help="dossier de sortie (defaut : a cote de chaque original)")
    g.add_argument("-r", "--recursif", action="store_true",
                   help="parcourir les sous-dossiers")
    g.add_argument("--suffixe", default="_naturaw",
                   help="suffixe ajoute au nom de fichier (defaut : %(default)s)")
    g.add_argument("-f", "--format", dest="fmt", default="tiff16",
                   choices=["tiff16", "tiff8", "png16", "png", "jpeg"],
                   help="format de sortie (defaut : %(default)s)")
    g.add_argument("-q", "--qualite", type=int, default=95,
                   help="qualite JPEG 1-100 (defaut : %(default)s)")
    g.add_argument("--courbe", default="srgb",
                   choices=["srgb", "rec709", "gamma22", "linear"],
                   help="courbe de transfert a l'ecriture (defaut : %(default)s)")
    g.add_argument("--ecraser", action="store_true",
                   help="ecraser un fichier de sortie existant (sinon, suffixe -2, -3...)")
    g.add_argument("--sans-dither", dest="dither", action="store_false",
                   help="desactiver le bruit de quantification des sorties 8 bits")

    g = p.add_argument_group("debruitage")
    g.add_argument("-d", "--debruitage", default="normal", metavar="NIVEAU",
                   help="off, leger, normal, fort, extreme, ou un nombre "
                        "(1 = nominal). Defaut : %(default)s")
    g.add_argument("--grain", type=float, default=DenoiseParams.grain,
                   help="part du grain fin conservee, 0-0.4 (defaut : %(default)s)")
    g.add_argument("--detail", type=float, default=DenoiseParams.detail,
                   help="protection des zones texturees, 0-1 (defaut : %(default)s)")
    g.add_argument("--chroma", type=float, default=DenoiseParams.chroma,
                   help="force relative sur la couleur (defaut : %(default)s)")
    g.add_argument("--echelles", type=int, default=0,
                   help="nombre d'echelles (0 = automatique)")
    g.add_argument("--nettoyage-couleur", type=float, default=DenoiseParams.chroma_cleanup,
                   help="dose de nettoyage des plaques de couleur, 0-1 (defaut : %(default)s)")

    g = p.add_argument_group("sauvetage (defauts du capteur)")
    g.add_argument("--pixels-chauds", default="auto", choices=["auto", "off"],
                   help="suppression des pixels chauds / morts (defaut : %(default)s)")
    g.add_argument("--bandes", default="off", choices=["off", "auto", "h", "v", "hv"],
                   help="correction des bandes de lecture du capteur ; a activer "
                        "si vous voyez des rayures (defaut : %(default)s)")

    g = p.add_argument_group("nettete et tonalite")
    g.add_argument("--nettete", type=float, default=SharpenParams.amount, metavar="DOSE",
                   help="nettete de capture, 0 a 1.5 (defaut : %(default)s)")
    g.add_argument("--nettete-rayon", type=float, default=SharpenParams.radius,
                   metavar="PX", help="douceur supposee du capteur (defaut : %(default)s)")
    g.add_argument("--ombres", default="auto", metavar="DOSE",
                   help="ouverture des ombres : auto, non, ou 0 a 1 "
                        "(1 ~ +3 EV dans les noirs). Defaut : %(default)s")
    g.add_argument("--recup-hautes-lumieres", type=float, default=0.0, metavar="DOSE",
                   help="recuperation des hautes lumieres, 0 a 1")
    g.add_argument("--contraste-local", type=float, default=0.0, metavar="DOSE",
                   help="renforcement du contraste local, 0 a 1")

    g = p.add_argument_group("deflou de bouge")
    g.add_argument("-b", "--deflou", default="auto",
                   choices=["off", "auto", "on", "force"],
                   help="off ; auto (seulement si bouge franc detecte) ; on "
                        "(ne garde que le test de nettete) ; force (aucun test). "
                        "Defaut : %(default)s")
    g.add_argument("--deflou-dose", type=float, default=DeblurParams.strength,
                   help="dose de correction, 0-1 (defaut : %(default)s)")
    g.add_argument("--deflou-iterations", type=int, default=DeblurParams.iterations,
                   help="iterations de Richardson-Lucy (defaut : %(default)s)")
    g.add_argument("--deflou-longueur", type=float, default=0.0, metavar="PX",
                   help="impose un bouge rectiligne de cette longueur en pixels")
    g.add_argument("--deflou-angle", type=float, default=0.0, metavar="DEG",
                   help="angle du bouge impose (degres)")
    g.add_argument("--deflou-noyau", type=int, default=31, metavar="PX",
                   help="taille du noyau estime (defaut : %(default)s)")
    g.add_argument("--deflou-noyau-libre", action="store_true",
                   help="deconvoluer avec le noyau libre estime au lieu de sa "
                        "projection rectiligne (utile pour un bouge en courbe)")
    g.add_argument("--deflou-canaux", default="luma", choices=["luma", "rgb"],
                   help="luma : transfert de nettete sans toucher aux couleurs "
                        "(defaut) ; rgb : deconvolution de chaque canal")

    g = p.add_argument_group("redressement")
    g.add_argument("-s", "--redressement", default="auto", choices=["auto", "off"],
                   help="redressement automatique (defaut : %(default)s)")
    g.add_argument("--angle", type=float, default=None, metavar="DEG",
                   help="angle de rotation impose (degres, sens trigonometrique)")
    g.add_argument("--angle-max", type=float, default=DeskewParams.max_angle,
                   help="correction maximale acceptee (defaut : %(default)s deg)")
    g.add_argument("--confiance", type=float, default=DeskewParams.min_confidence,
                   help="confiance minimale pour redresser (defaut : %(default)s)")
    g.add_argument("--recadrage", default="aspect", choices=["aspect", "max", "none"],
                   help="aspect : garde le format ; max : aire maximale ; "
                        "none : aucun recadrage (defaut : %(default)s)")

    g = p.add_argument_group("developpement du RAW")
    g.add_argument("--dematricage", default="ahd",
                   choices=["ahd", "vng", "ppg", "dcb", "dht", "aahd", "linear"],
                   help="algorithme de dematricage (defaut : %(default)s)")
    g.add_argument("--balance", default="boitier", choices=["boitier", "auto"],
                   help="balance des blancs (defaut : %(default)s)")
    g.add_argument("--espace", default="srgb",
                   choices=["srgb", "adobe", "prophoto", "wide", "xyz", "raw"],
                   help="espace colorimetrique de sortie (defaut : %(default)s ; "
                        "aucun profil ICC n'est incorpore, les espaces autres que "
                        "sRGB sont pour qui gere sa couleur lui-meme)")
    g.add_argument("--hautes-lumieres", type=int, default=0, metavar="N",
                   help="0 = ecreter, 1 = conserver, 2 = fondre, 3+ = reconstruire")
    g.add_argument("--demi-taille", action="store_true",
                   help="developper en demi-resolution (tres rapide)")
    g.add_argument("--cote-max", type=int, default=0, metavar="PX",
                   help="reduire l'image a ce plus grand cote avant traitement")
    g.add_argument("--exposition", default="auto", metavar="MODE",
                   help="auto, non, ou une correction en EV (ex. +0.7). "
                        "Defaut : %(default)s")

    g = p.add_argument_group("divers")
    g.add_argument("-j", "--jobs", type=int, default=1,
                   help="nombre de fichiers traites en parallele (defaut : %(default)s)")
    g.add_argument("--apercu", action="store_true",
                   help="apercu de controle : l'image entiere reduite, plus une "
                        "zone a 100 %% traitee exactement comme le fichier final "
                        "(c'est la qu'on juge le bruit et la nettete)")
    g.add_argument("--rapide", action="store_true",
                   help="priorite a la vitesse : une seule passe de debruitage "
                        "et pas d'analyse de bouge (environ deux fois plus rapide)")
    g.add_argument("--rapport", metavar="FICHIER", default="",
                   help="ecrire un rapport JSON detaille")
    g.add_argument("--interface", nargs="?", const=8731, type=int, metavar="PORT",
                   help="lancer l'interface web locale (defaut : port 8731)")
    g.add_argument("--liste", action="store_true",
                   help="lister les fichiers trouves sans rien traiter")
    g.add_argument("-v", "--verbeux", action="store_true", help="details de traitement")
    g.add_argument("--version", action="version", version=f"naturaw {__version__}")
    return p


def settings_from_args(a: argparse.Namespace) -> Settings:
    """Construit les reglages a partir des arguments analyses."""
    level = str(a.debruitage).lower()
    try:
        strength = PRESETS[level] if level in PRESETS else float(level)
    except ValueError:
        raise SystemExit(f"niveau de debruitage invalide : {a.debruitage!r} "
                         f"(attendu : {', '.join(PRESETS)} ou un nombre)")
    denoise = DenoiseParams(
        strength=strength,
        chroma=a.chroma,
        grain=a.grain,
        detail=a.detail,
        levels=a.echelles,
        chroma_cleanup=getattr(a, "nettoyage_couleur"),
    )
    deblur = DeblurParams(
        mode=a.deflou,
        strength=a.deflou_dose,
        iterations=a.deflou_iterations,
        kernel_size=a.deflou_noyau,
        channels=a.deflou_canaux,
        length=a.deflou_longueur,
        angle=a.deflou_angle,
        free_kernel=getattr(a, "deflou_noyau_libre"),
    )
    tone_shadows = 0.0
    tone_auto = True
    mode_ombres = str(a.ombres).lower()
    if mode_ombres in ("non", "off", "none", "0"):
        tone_auto = False
    elif mode_ombres not in ("auto", "oui", "yes"):
        try:
            tone_shadows = float(mode_ombres)
            tone_auto = False
        except ValueError:
            raise SystemExit(f"valeur invalide pour --ombres : {a.ombres!r} "
                             "(attendu : auto, non, ou un nombre entre 0 et 1)")
    tone = ToneParams(shadows=tone_shadows, auto=tone_auto,
                      highlights=getattr(a, "recup_hautes_lumieres"),
                      local_contrast=a.contraste_local)
    rescue = RescueParams(impulse=getattr(a, "pixels_chauds"), banding=a.bandes)
    sharpen = SharpenParams(amount=a.nettete, radius=a.nettete_rayon)
    deskew = DeskewParams(
        enabled=(a.redressement != "off") or a.angle is not None,
        max_angle=a.angle_max,
        min_confidence=a.confiance,
        crop=a.recadrage,
        angle=a.angle,
    )
    s = Settings(
        rescue=rescue, denoise=denoise, deblur=deblur, sharpen=sharpen,
        tone=tone, deskew=deskew,
        demosaic=a.dematricage,
        camera_wb=(a.balance == "boitier"),
        color_space=a.espace,
        highlight=getattr(a, "hautes_lumieres"),
        half_size=a.demi_taille,
        max_side=a.cote_max,
        fmt=a.fmt, quality=a.qualite, transfer=a.courbe,
        exposure=a.exposition, dither=a.dither,
        suffix=a.suffixe, overwrite=a.ecraser,
    )
    if a.rapide:
        s.denoise.refine = False
        s.deblur.mode = "off"
    if a.apercu:
        s.max_side = s.max_side or 1800
        s.fmt = "jpeg" if a.fmt == "tiff16" else s.fmt
        s.suffix = a.suffixe if a.suffixe != "_naturaw" else "_apercu"
    return s


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    a = parser.parse_args(argv)

    if a.interface is not None:
        from .webui import serve
        return serve(port=int(a.interface))

    if not a.entrees:
        parser.print_help()
        return 2

    files = collect_inputs(a.entrees, recursive=a.recursif)
    if not files:
        print("Aucun fichier RAW ou image reconnu dans ce qui a ete indique.",
              file=sys.stderr)
        return 1
    if a.liste:
        for f in files:
            print(f)
        return 0

    settings = settings_from_args(a)
    if a.sortie:
        os.makedirs(a.sortie, exist_ok=True)

    total = len(files)
    print(f"naturaw {__version__} - {total} fichier(s) a traiter"
          f"{' (apercu rapide)' if a.apercu else ''}")

    def on_done(done: int, count: int, rep) -> None:
        print(f"[{done}/{count}] {summarize(rep)}", flush=True)

    reports = process_many(files, a.sortie, settings, jobs=a.jobs, progress=on_done,
                           preview=a.apercu)

    failed = [r for r in reports if "erreur" in r]
    if a.verbeux:
        for r in reports:
            print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
    if a.rapport:
        with open(a.rapport, "w", encoding="utf-8") as fh:
            json.dump(reports, fh, ensure_ascii=False, indent=2, default=str)
        print(f"Rapport ecrit dans {a.rapport}")
    print(f"Termine : {total - len(failed)} reussite(s), {len(failed)} echec(s).")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
