"""Permet ``python -m naturaw``.

Le garde ``__main__`` n'est pas decoratif : sans lui, les processus de
travail demarres en mode ``spawn`` reexecuteraient ce fichier et relanceraient
le programme en boucle.
"""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
