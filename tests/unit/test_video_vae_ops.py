import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from lara_ltx.errors import LaraError
from lara_ltx.video_vae.ops import PerChannelStatistics, patchify, unpatchify

PATCH_SIZE_HW = 2
PATCH_SIZE_T = 2


def test_patchify_and_unpatchify_are_inverse_for_video_layout() -> None:
    source = mx.arange(1 * 3 * 4 * 4 * 6, dtype=mx.float32).reshape(1, 3, 4, 4, 6)
    patched = patchify(source, patch_size_hw=PATCH_SIZE_HW, patch_size_t=PATCH_SIZE_T)
    restored = unpatchify(patched, patch_size_hw=PATCH_SIZE_HW, patch_size_t=PATCH_SIZE_T)
    np.testing.assert_array_equal(np.asarray(restored), np.asarray(source))


def test_per_channel_statistics_normalization_round_trip() -> None:
    statistics = PerChannelStatistics(latent_channels=2)
    statistics.load_weights(
        [
            ("std_of_means", mx.array([2.0, 4.0])),
            ("mean_of_means", mx.array([1.0, -3.0])),
        ]
    )
    source = mx.array([[[[[5.0]]], [[[-7.0]]]]])
    np.testing.assert_allclose(
        np.asarray(statistics.un_normalize(statistics.normalize(source))),
        np.asarray(source),
    )


def test_patchify_rejects_indivisible_layout() -> None:
    with pytest.raises(LaraError, match="LARA-TENSOR-012"):
        patchify(mx.zeros((1, 1, 3, 4, 4)), patch_size_hw=PATCH_SIZE_HW, patch_size_t=PATCH_SIZE_T)
