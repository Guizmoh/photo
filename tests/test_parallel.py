"""Le traitement par lot en parallele ne doit pas se bloquer.

Sur Linux, multiprocessing duplique le processus avec ``fork``. Si le parent
a deja des fils d'execution -- et c'est le cas des que OpenCV ou l'OpenMP de
LibRaw ont servi -- les verrous sont copies dans un etat indetermine et les
processus enfants se figent pour toujours. rawpy previent lui-meme du
probleme. Ce test l'attrape : il s'execute dans un sous-processus avec un
delai maximal, pour qu'un blocage fasse echouer la suite au lieu de la figer.
"""

import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPT = r'''
import os, sys, tempfile
sys.path.insert(0, {root!r}); sys.path.insert(0, os.path.join({root!r}, "tests"))
import synth
from naturaw.pipeline import Settings, process_many

def run():
    d = tempfile.mkdtemp()
    scene = synth.architecture(240, 360)
    paths = []
    for i in range(3):
        p = os.path.join(d, "f%d.dng" % i)
        synth.write_dng_mosaic(p, synth.simulate_raw_capture(scene, 600.0, 3.0, seed=i))
        paths.append(p)
    s = Settings(fmt="jpeg", color_space="raw", max_side=300)
    # un premier passage sequentiel cree les fils : c'est la situation qui
    # provoquait le blocage
    process_many(paths[:1], d, s, jobs=1)
    reports = process_many(paths, d, s, jobs=2)
    assert len(reports) == 3, reports
    assert not any("erreur" in r for r in reports), reports
    assert all(os.path.exists(r["sortie"]) for r in reports)
    print("OK")

if __name__ == "__main__":
    run()
'''


def test_batch_in_parallel_does_not_deadlock(tmp_path):
    script = tmp_path / "lot.py"
    script.write_text(SCRIPT.format(root=ROOT), encoding="utf-8")
    try:
        done = subprocess.run([sys.executable, str(script)], capture_output=True,
                              text=True, timeout=240)
    except subprocess.TimeoutExpired:
        pytest.fail("le traitement parallele s'est bloque (fork + OpenMP)")
    assert done.returncode == 0, done.stderr[-2000:]
    assert "OK" in done.stdout


DETERMINISM = r"""
import hashlib, os, sys, tempfile
sys.path.insert(0, {root!r}); sys.path.insert(0, os.path.join({root!r}, "tests"))
import synth
from naturaw.pipeline import Settings, process_many

def digest(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()

def run():
    d = tempfile.mkdtemp()
    scene = synth.architecture(400, 600)
    paths = []
    for i in range(2):
        p = os.path.join(d, "d%d.dng" % i)
        synth.write_dng_mosaic(p, synth.simulate_raw_capture(scene, 600.0, 3.0, seed=i))
        paths.append(p)
    s = Settings(fmt="png16", color_space="raw")
    sigs = []
    for jobs in (1, 2):
        out = os.path.join(d, "o%d" % jobs)
        os.makedirs(out, exist_ok=True)
        sigs.append([digest(r["sortie"]) for r in process_many(paths, out, s, jobs=jobs)])
    assert sigs[0] == sigs[1], sigs
    print("OK")

if __name__ == "__main__":
    run()
"""


def test_results_do_not_depend_on_the_number_of_processes(tmp_path):
    """Le meme fichier doit donner le meme resultat, seul ou en parallele."""
    script = tmp_path / "det.py"
    script.write_text(DETERMINISM.format(root=ROOT), encoding="utf-8")
    try:
        done = subprocess.run([sys.executable, str(script)], capture_output=True,
                              text=True, timeout=240)
    except subprocess.TimeoutExpired:
        pytest.fail("le traitement parallele s'est bloque")
    assert done.returncode == 0, done.stderr[-2000:]
    assert "OK" in done.stdout


def test_pool_context_avoids_fork():
    from naturaw.pipeline import _pool_context
    ctx = _pool_context()
    assert ctx is not None
    assert ctx.get_start_method() in ("forkserver", "spawn")


def test_auto_jobs_is_sane(tmp_path):
    """Le choix automatique doit rester dans les clous : coeurs et memoire."""
    from naturaw.pipeline import auto_jobs
    n = auto_jobs()
    assert 1 <= n <= max(1, os.cpu_count() or 2)
    # un fichier enorme doit faire baisser le nombre de processus
    big = tmp_path / "enorme.dng"
    big.write_bytes(b"\0" * (400 * 1024 * 1024))     # ~330 Mpx estimes
    assert auto_jobs([str(big)]) <= n


def test_entry_point_is_guarded():
    """``python -m naturaw`` ne doit pas se relancer dans les sous-processus."""
    source = open(os.path.join(ROOT, "naturaw", "__main__.py"), encoding="utf-8").read()
    assert '__name__ == "__main__"' in source
