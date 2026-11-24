"""Reprise des metadonnees de la prise de vue.

Une photo developpee qui a perdu sa date, son boitier et son objectif est
difficile a classer : la plupart des logiciels de catalogage trient sur ces
champs. naturaw relit donc l'EXIF du fichier d'origine et le recopie dans le
fichier produit -- la date de prise de vue, le materiel, les parametres
d'exposition et, s'il y en a, la position GPS.

Deux bibliotheques facultatives interviennent : ``exifread`` pour lire (elle
comprend les RAW de la plupart des marques) et ``piexif`` pour ecrire le bloc
EXIF d'un JPEG. Si elles manquent, le traitement se fait quand meme, sans les
metadonnees : rien n'echoue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

# Champs repris, avec le nom tel qu'exifread le donne
_WANTED = {
    "marque": "Image Make",
    "boitier": "Image Model",
    "objectif": "EXIF LensModel",
    "date": "EXIF DateTimeOriginal",
    "date_numerisation": "EXIF DateTimeDigitized",
    "iso": "EXIF ISOSpeedRatings",
    "vitesse": "EXIF ExposureTime",
    "ouverture": "EXIF FNumber",
    "focale": "EXIF FocalLength",
    "focale_35": "EXIF FocalLengthIn35mmFilm",
    "auteur": "Image Artist",
    "droits": "Image Copyright",
    "correction": "EXIF ExposureBiasValue",
}
_GPS = {
    "gps_lat": "GPS GPSLatitude",
    "gps_lat_ref": "GPS GPSLatitudeRef",
    "gps_lon": "GPS GPSLongitude",
    "gps_lon_ref": "GPS GPSLongitudeRef",
    "gps_alt": "GPS GPSAltitude",
}


@dataclass
class CaptureInfo:
    """Ce que l'on sait de la prise de vue."""

    fields: Dict[str, Any] = field(default_factory=dict)
    gps: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.fields or self.gps)

    def summary(self) -> str:
        f = self.fields
        bits = []
        if f.get("boitier"):
            bits.append(str(f.get("marque", "")).strip() + " " + str(f["boitier"]).strip())
        if f.get("objectif"):
            bits.append(str(f["objectif"]).strip())
        if f.get("vitesse"):
            bits.append(f"{f['vitesse']} s")
        if f.get("ouverture"):
            bits.append(f"f/{_as_float(f['ouverture']):.1f}")
        if f.get("iso"):
            bits.append(f"{f['iso']} ISO")
        if f.get("date"):
            bits.append(str(f["date"]))
        if self.gps:
            bits.append("position GPS")
        return "  ".join(b for b in bits if b.strip())


def _as_float(value: Any) -> float:
    try:
        if hasattr(value, "num") and hasattr(value, "den"):
            return float(value.num) / float(value.den or 1)
        text = str(value)
        if "/" in text:
            a, b = text.split("/", 1)
            return float(a) / float(b or 1)
        return float(text)
    except (ValueError, TypeError, ZeroDivisionError):
        return 0.0


def _as_rational(value: Any, precision: int = 1000) -> Optional[Tuple[int, int]]:
    """Fraction entiere, telle que l'EXIF les stocke."""
    try:
        if hasattr(value, "num") and hasattr(value, "den") and value.den:
            return int(value.num), int(value.den)
        text = str(value).strip()
        if "/" in text:
            a, b = text.split("/", 1)
            return int(float(a)), int(float(b))
        f = float(text)
        if f <= 0:
            return None
        return int(round(f * precision)), precision
    except (ValueError, TypeError):
        return None


def read_capture_info(path: str) -> CaptureInfo:
    """Lit les champs utiles du fichier d'origine (RAW, JPEG, TIFF)."""
    try:
        import exifread
    except ImportError:
        return CaptureInfo()
    try:
        with open(path, "rb") as fh:
            tags = exifread.process_file(fh, details=False)
    except Exception:
        return CaptureInfo()
    info = CaptureInfo()
    for key, name in _WANTED.items():
        if name in tags:
            raw = tags[name]
            values = getattr(raw, "values", None)
            if isinstance(values, list) and len(values) == 1:
                info.fields[key] = values[0]
            else:
                info.fields[key] = str(raw)
    for key, name in _GPS.items():
        if name in tags:
            info.gps[key] = tags[name]
    return info


def _gps_block(gps: Dict[str, Any]):
    """Bloc GPS au format piexif, ou None."""
    try:
        import piexif
    except ImportError:
        return None
    out: Dict[int, Any] = {}

    def dms(tag):
        values = getattr(gps.get(tag), "values", None)
        if not values or len(values) != 3:
            return None
        return tuple((int(v.num), int(v.den or 1)) for v in values)

    lat, lon = dms("gps_lat"), dms("gps_lon")
    if lat and lon:
        out[piexif.GPSIFD.GPSLatitude] = lat
        out[piexif.GPSIFD.GPSLongitude] = lon
        out[piexif.GPSIFD.GPSLatitudeRef] = str(gps.get("gps_lat_ref", "N"))[:1] or "N"
        out[piexif.GPSIFD.GPSLongitudeRef] = str(gps.get("gps_lon_ref", "E"))[:1] or "E"
    alt = _as_rational(getattr(gps.get("gps_alt"), "values", [None])[0]
                       if gps.get("gps_alt") is not None else None)
    if alt:
        out[piexif.GPSIFD.GPSAltitude] = alt
    return out or None


def exif_payload(info: CaptureInfo, software: str = "naturaw") -> Optional[bytes]:
    """Construit un bloc EXIF prêt a inserer dans un JPEG, ou None."""
    try:
        import piexif
    except ImportError:
        return None
    f = info.fields
    zeroth: Dict[int, Any] = {piexif.ImageIFD.Software: software,
                              piexif.ImageIFD.Orientation: 1}
    exif: Dict[int, Any] = {}
    text_map = {
        "marque": (zeroth, piexif.ImageIFD.Make),
        "boitier": (zeroth, piexif.ImageIFD.Model),
        "auteur": (zeroth, piexif.ImageIFD.Artist),
        "droits": (zeroth, piexif.ImageIFD.Copyright),
        "objectif": (exif, piexif.ExifIFD.LensModel),
        "date": (exif, piexif.ExifIFD.DateTimeOriginal),
        "date_numerisation": (exif, piexif.ExifIFD.DateTimeDigitized),
    }
    for key, (block, tag) in text_map.items():
        value = f.get(key)
        if value is not None and str(value).strip():
            block[tag] = str(value).strip()[:120]
    if f.get("date"):
        zeroth[piexif.ImageIFD.DateTime] = str(f["date"]).strip()[:20]
    iso = f.get("iso")
    if iso is not None:
        try:
            exif[piexif.ExifIFD.ISOSpeedRatings] = int(_as_float(iso))
        except (ValueError, TypeError):
            pass
    for key, tag in (("vitesse", piexif.ExifIFD.ExposureTime),
                     ("ouverture", piexif.ExifIFD.FNumber),
                     ("focale", piexif.ExifIFD.FocalLength)):
        rational = _as_rational(f.get(key)) if f.get(key) is not None else None
        if rational and rational[1]:
            exif[tag] = rational
    if f.get("focale_35"):
        try:
            exif[piexif.ExifIFD.FocalLengthIn35mmFilm] = int(_as_float(f["focale_35"]))
        except (ValueError, TypeError):
            pass
    blocks: Dict[str, Any] = {"0th": zeroth, "Exif": exif, "1st": {}, "thumbnail": None}
    gps = _gps_block(info.gps)
    if gps:
        blocks["GPS"] = gps
    try:
        return piexif.dump(blocks)
    except Exception:
        return None


def attach_to_jpeg(path: str, payload: Optional[bytes]) -> bool:
    """Insere un bloc EXIF dans un JPEG deja ecrit. Renvoie True si c'est fait."""
    if not payload:
        return False
    try:
        import piexif
        piexif.insert(payload, path)
        return True
    except Exception:
        return False


def tiff_extratags(info: CaptureInfo, software: str = "naturaw"):
    """Etiquettes TIFF standard a ecrire avec tifffile."""
    f = info.fields
    # tifffile ecrit lui-meme l'etiquette Software (305) : on ne la double pas
    tags = []
    mapping = {"marque": 271, "boitier": 272, "auteur": 315, "droits": 33432}
    for key, code in mapping.items():
        value = f.get(key)
        if value is not None and str(value).strip():
            # le TIFF n'accepte que de l'ASCII 7 bits
            text = str(value).strip().encode("ascii", "replace").decode("ascii")
            tags.append((code, "s", 0, text[:120], True))
    if f.get("date"):
        text = str(f["date"]).strip()[:19]
        tags.append((306, "s", 0, text, True))      # DateTime
    return tags
