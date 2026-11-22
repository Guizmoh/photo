"""naturaw - debruitage naturel, deflou de bouge et redressement de RAW.

Exemple d'utilisation comme bibliotheque :

    from naturaw import Settings, process_file
    report = process_file("photo.CR2", "photo.tif", Settings())
"""

__version__ = "0.1.0"

from .colorspace import linear_to_srgb, srgb_to_linear           # noqa: F401
from .deblur import DeblurParams, analyze_blur, deblur_linear_rgb  # noqa: F401
from .denoise import DenoiseParams, denoise_linear_rgb            # noqa: F401
from .deskew import DeskewParams, deskew_image, estimate_tilt     # noqa: F401
from .noise import measure_profile                                # noqa: F401
from .pipeline import (Settings, collect_inputs, process_array,    # noqa: F401
                       process_file, process_many, summarize)
from .rawio import load_image, save_image                         # noqa: F401

__all__ = [
    "__version__", "Settings", "process_file", "process_array", "process_many",
    "collect_inputs", "summarize", "DenoiseParams", "DeblurParams", "DeskewParams",
    "denoise_linear_rgb", "deblur_linear_rgb", "deskew_image", "estimate_tilt",
    "analyze_blur", "measure_profile", "load_image", "save_image",
    "linear_to_srgb", "srgb_to_linear",
]
