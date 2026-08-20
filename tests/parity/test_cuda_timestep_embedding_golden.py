from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from lara_ltx.transformer.timestep import PixArtAlphaCombinedTimestepSizeEmbeddings

GOLDEN_PATH = Path("golden/cuda/timestep_embedding_bf16_cuda.npz")
EMBEDDING_DIM = 16
BF16_ABSOLUTE_TOLERANCE = 2e-2
BF16_RELATIVE_TOLERANCE = 2e-2
WEIGHT_PREFIX = "timestep_embedding_weight__"


def test_pixart_timestep_embedding_matches_cuda_bf16() -> None:
    if not GOLDEN_PATH.is_file():
        pytest.skip("CUDA timestep embedding golden artifact has not been generated")
    with np.load(GOLDEN_PATH) as archive:
        golden = {key: archive[key] for key in archive.files}
    embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(EMBEDDING_DIM)
    embedder.load_weights(
        [
            (name.removeprefix(WEIGHT_PREFIX), mx.array(value, dtype=mx.bfloat16))
            for name, value in golden.items()
            if name.startswith(WEIGHT_PREFIX)
        ]
    )
    result = embedder(mx.array(golden["timestep"], dtype=mx.bfloat16), hidden_dtype=mx.bfloat16)
    np.testing.assert_allclose(
        np.asarray(result.astype(mx.float32)),
        golden["output"],
        rtol=BF16_RELATIVE_TOLERANCE,
        atol=BF16_ABSOLUTE_TOLERANCE,
    )
