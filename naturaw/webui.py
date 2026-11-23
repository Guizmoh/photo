"""Interface graphique locale (serveur web de la bibliotheque standard).

Lancee par ``naturaw --interface``, elle sert une page unique sur
``127.0.0.1`` : parcours des dossiers, selection des photos, reglages avec
apercu avant/apres, traitement par lot avec avancement. Aucune dependance
supplementaire, aucun acces reseau exterieur : le serveur n'ecoute que sur la
machine locale.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
import webbrowser
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from . import __version__
from .deblur import DeblurParams
from .denoise import PRESETS, DenoiseParams
from .deskew import DeskewParams
from .rescue import RescueParams
from .sharpen import SharpenParams
from .tone import ToneParams
from .pipeline import Settings, process_file, summarize
from .rawio import RAW_EXTENSIONS, IMAGE_EXTENSIONS, default_output_name

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "web", "index.html")

_JOBS: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()
PREVIEW_DIR = os.path.join(os.path.expanduser("~"), ".cache", "naturaw", "apercus")


def _settings_from_payload(p: Dict[str, Any]) -> Settings:
    """Construit des reglages a partir du JSON envoye par la page."""
    level = str(p.get("debruitage", "normal"))
    strength = PRESETS.get(level.lower(), None)
    if strength is None:
        try:
            strength = float(level)
        except ValueError:
            strength = 1.0
    denoise = DenoiseParams(
        strength=float(strength),
        chroma=float(p.get("chroma", DenoiseParams.chroma)),
        grain=float(p.get("grain", DenoiseParams.grain)),
        detail=float(p.get("detail", DenoiseParams.detail)),
        chroma_cleanup=float(p.get("nettoyage", DenoiseParams.chroma_cleanup)),
    )
    deblur = DeblurParams(
        mode=str(p.get("deflou", "auto")),
        strength=float(p.get("deflou_dose", DeblurParams.strength)),
        length=float(p.get("deflou_longueur", 0.0) or 0.0),
        angle=float(p.get("deflou_angle", 0.0) or 0.0),
        free_kernel=bool(p.get("deflou_noyau_libre", False)),
    )
    deskew = DeskewParams(
        enabled=bool(p.get("redressement", True)),
        max_angle=float(p.get("angle_max", DeskewParams.max_angle)),
        min_confidence=float(p.get("confiance", DeskewParams.min_confidence)),
        crop=str(p.get("recadrage", "aspect")),
    )
    rescue = RescueParams(
        impulse="auto" if p.get("pixels_chauds", True) else "off",
        banding=str(p.get("bandes", "off")),
    )
    sharpen = SharpenParams(amount=float(p.get("nettete", SharpenParams.amount)))
    ombres = str(p.get("ombres", "auto"))
    if ombres in ("auto", ""):
        tone = ToneParams(auto=True)
    elif ombres in ("non", "off"):
        tone = ToneParams(auto=False)
    else:
        tone = ToneParams(auto=False, shadows=float(ombres))
    tone.highlights = float(p.get("hautes_lumieres", 0.0))
    tone.local_contrast = float(p.get("contraste_local", 0.0))
    if p.get("rapide"):
        denoise.refine = False
        deblur.mode = "off"
    return Settings(
        rescue=rescue, denoise=denoise, deblur=deblur, sharpen=sharpen,
        tone=tone, deskew=deskew,
        fmt=str(p.get("format", "tiff16")),
        quality=int(p.get("qualite", 95)),
        exposure=str(p.get("exposition", "auto")),
        max_side=int(p.get("cote_max", 0) or 0),
        half_size=bool(p.get("demi_taille", False)),
        suffix=str(p.get("suffixe", "_naturaw")),
    )


def _write_preview(path: str, settings: Settings, job_id: str, tag: str) -> Optional[str]:
    """Genere un couple d'apercus avant/apres (JPEG, cote max 1500 px)."""
    import cv2
    import numpy as np

    from .colorspace import linear_to_srgb
    from .rawio import load_image

    os.makedirs(PREVIEW_DIR, exist_ok=True)
    try:
        loaded = load_image(path, half_size=True)
    except Exception:
        return None
    rgb = loaded.rgb
    long_side = max(rgb.shape[:2])
    if long_side > 1500:
        sc = 1500.0 / long_side
        rgb = cv2.resize(rgb, (int(rgb.shape[1] * sc), int(rgb.shape[0] * sc)),
                         interpolation=cv2.INTER_AREA)
    from .pipeline import process_array
    from .rawio import auto_exposure
    gain = auto_exposure(rgb) if settings.exposure in ("auto", "oui") else 1.0
    before = linear_to_srgb(np.clip(rgb * gain, 0, 1))
    rep: Dict[str, Any] = {}
    after_lin = process_array(rgb.copy(), settings, report=rep)
    after = linear_to_srgb(np.clip(after_lin, 0, 1))
    out = {}
    for name, img in (("avant", before), ("apres", after)):
        data = (np.clip(img, 0, 1) * 255.0 + 0.5).astype("uint8")
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(data, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 88])
        if ok:
            fn = os.path.join(PREVIEW_DIR, f"{job_id}_{tag}_{name}.jpg")
            with open(fn, "wb") as fh:
                fh.write(buf.tobytes())
            out[name] = fn
    return json.dumps({"fichiers": out, "rapport": rep}, default=str)


def _run_job(job_id: str, files: List[str], out_dir: str, settings: Settings,
             preview_only: bool) -> None:
    job = _JOBS[job_id]
    try:
        for i, path in enumerate(files):
            with _LOCK:
                job["courant"] = os.path.basename(path)
                job["avancement"] = i / max(1, len(files))
            if preview_only:
                res = _write_preview(path, settings, job_id, str(i))
                with _LOCK:
                    if res:
                        payload = json.loads(res)
                        job["apercus"].append({"fichier": path, **payload})
                        job["rapports"].append({"entree": path, **payload["rapport"]})
                continue
            name = default_output_name(path, settings.suffix, settings.fmt)
            target = os.path.join(out_dir or os.path.dirname(os.path.abspath(path)), name)
            try:
                rep = process_file(path, target, settings)
            except Exception as exc:
                rep = {"entree": path, "erreur": f"{type(exc).__name__}: {exc}"}
            with _LOCK:
                job["rapports"].append(rep)
                job["resume"].append(summarize(rep))
    finally:
        with _LOCK:
            job["avancement"] = 1.0
            job["termine"] = True
            job["courant"] = ""


class Handler(BaseHTTPRequestHandler):
    server_version = f"naturaw/{__version__}"

    def log_message(self, fmt, *args):  # silence : la console sert au traitement
        pass

    # ------------------------------------------------------------- utilitaires
    def _json(self, obj: Any, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: str, ctype: str) -> None:
        try:
            with open(path, "rb") as fh:
                body = fh.read()
        except OSError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------------- GET
    def do_GET(self) -> None:
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            self._file(PAGE, "text/html; charset=utf-8")
            return
        if u.path == "/api/defauts":
            self._json({
                "version": __version__,
                "dossier": os.path.expanduser("~"),
                "debruitage": list(PRESETS),
                "denoise": asdict(DenoiseParams()),
                "deblur": asdict(DeblurParams()),
                "deskew": {k: v for k, v in asdict(DeskewParams()).items()
                           if k != "angle"},
            })
            return
        if u.path == "/api/parcourir":
            d = os.path.abspath(os.path.expanduser(q.get("dir", [os.path.expanduser("~")])[0]))
            if not os.path.isdir(d):
                self._json({"erreur": f"dossier introuvable : {d}"}, 404)
                return
            dirs, files = [], []
            try:
                for name in sorted(os.listdir(d), key=str.lower):
                    full = os.path.join(d, name)
                    if name.startswith("."):
                        continue
                    if os.path.isdir(full):
                        dirs.append(name)
                    elif os.path.splitext(name)[1].lower() in (RAW_EXTENSIONS | IMAGE_EXTENSIONS):
                        files.append({"nom": name,
                                      "taille": os.path.getsize(full),
                                      "raw": os.path.splitext(name)[1].lower() in RAW_EXTENSIONS})
            except PermissionError:
                self._json({"erreur": "acces refuse"}, 403)
                return
            self._json({"dossier": d, "parent": os.path.dirname(d) or d,
                        "dossiers": dirs, "fichiers": files})
            return
        if u.path == "/api/tache":
            jid = q.get("id", [""])[0]
            with _LOCK:
                job = _JOBS.get(jid)
                self._json(job or {"erreur": "tache inconnue"}, 200 if job else 404)
            return
        if u.path == "/api/image":
            path = q.get("path", [""])[0]
            if not path.startswith(PREVIEW_DIR) or not os.path.isfile(path):
                self.send_error(403)
                return
            self._file(path, "image/jpeg")
            return
        self.send_error(404)

    # ------------------------------------------------------------------ POST
    def do_POST(self) -> None:
        u = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json({"erreur": "JSON invalide"}, 400)
            return
        if u.path not in ("/api/traiter", "/api/apercu"):
            self.send_error(404)
            return
        files = [os.path.abspath(os.path.expanduser(f)) for f in payload.get("fichiers", [])]
        files = [f for f in files if os.path.isfile(f)]
        if not files:
            self._json({"erreur": "aucun fichier valide selectionne"}, 400)
            return
        settings = _settings_from_payload(payload.get("reglages", {}))
        out_dir = payload.get("sortie", "") or ""
        if out_dir:
            out_dir = os.path.abspath(os.path.expanduser(out_dir))
            os.makedirs(out_dir, exist_ok=True)
        preview = u.path == "/api/apercu"
        if preview:
            files = files[:1]
        jid = uuid.uuid4().hex[:12]
        with _LOCK:
            _JOBS[jid] = {"id": jid, "total": len(files), "avancement": 0.0,
                          "termine": False, "courant": "", "rapports": [],
                          "resume": [], "apercus": [], "apercu": preview,
                          "debut": time.time()}
        threading.Thread(target=_run_job, args=(jid, files, out_dir, settings, preview),
                         daemon=True).start()
        self._json({"id": jid, "total": len(files)})


def serve(port: int = 8731, open_browser: bool = True) -> int:
    """Demarre l'interface locale. Renvoie un code de sortie."""
    if not os.path.isfile(PAGE):
        print(f"Page introuvable : {PAGE}")
        return 1
    address = ("127.0.0.1", int(port))
    try:
        httpd = ThreadingHTTPServer(address, Handler)
    except OSError as exc:
        print(f"Impossible d'ouvrir le port {port} : {exc}")
        return 1
    url = f"http://127.0.0.1:{port}/"
    print(f"naturaw {__version__} - interface locale sur {url}")
    print("(Ctrl+C pour arreter ; le serveur n'ecoute que sur cette machine.)")
    if open_browser:
        threading.Thread(target=lambda: (time.sleep(0.6), webbrowser.open(url)),
                         daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nArret.")
    finally:
        httpd.server_close()
    return 0
