import numpy as np
import pytest

import synth
from naturaw.pyramid import (analyse, gaussian_stack, match_size,
                             pyramid_levels, synthese)


@pytest.mark.parametrize("shape", [(64, 96), (301, 457), (640, 960), (1200, 1800)])
def test_reconstruction_is_exact(shape):
    img = synth.architecture(*shape)[..., 1].copy()
    levels = pyramid_levels(shape)
    details, residual = analyse(img, levels)
    assert np.abs(synthese(details, residual) - img).max() < 1e-5


def test_levels_shrink_by_two():
    img = synth.architecture(512, 768)[..., 0].copy()
    details, residual = analyse(img, 4)
    assert details[0].shape == (512, 768)
    for a, b in zip(details, details[1:]):
        assert b.shape[0] == (a.shape[0] + 1) // 2


def test_level_count_follows_image_size():
    assert pyramid_levels((300, 400)) == 3
    assert pyramid_levels((4000, 6000)) == 6
    assert pyramid_levels((4000, 6000), requested=3) == 3
    assert pyramid_levels((40, 60)) >= 2       # ne descend jamais trop bas


def test_gaussian_stack_matches_detail_shapes():
    img = synth.architecture(400, 600)[..., 0].copy()
    details, _res = analyse(img, 4)
    stack = gaussian_stack(img, 4)
    for j, d in enumerate(details):
        assert match_size(stack[j], d.shape).shape == d.shape


def test_zero_image_stays_zero():
    z = np.zeros((128, 128), np.float32)
    details, residual = analyse(z, 3)
    assert all(np.all(d == 0) for d in details) and np.all(residual == 0)
