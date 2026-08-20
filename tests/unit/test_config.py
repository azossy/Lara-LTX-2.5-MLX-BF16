from pathlib import Path

from lara_ltx.config import load_project_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "project.toml"


def test_reference_configuration_is_quality_preserving() -> None:
    config = load_project_config(CONFIG_PATH)

    assert config.dtype == "bfloat16"
    assert config.quantization == "none"
    assert len(config.source_commit) == 40
    assert len(config.model_revision) == 40
    assert config.canonical_hq.output_height == 1_088
    assert config.canonical_hq.output_width == 1_920
    assert config.canonical_hq.num_frames == 121
    assert config.memory.minimum_system_reserve_bytes == 16 * 1024**3
    assert 0.0 < config.memory.maximum_peak_fraction < 1.0
