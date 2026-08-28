import sys
from pathlib import Path
from typing import cast

from omegaconf import DictConfig, OmegaConf

from asr.models import FastConformerRNNT, StreamingFastConformerRNNT
from asr.modules.conformer import FastConformer, StreamingFastConformer

EXPERIMENT_DIR = Path(__file__).resolve().parents[2] / "experiments" / "librispeech"
SCRIPTS_DIR = EXPERIMENT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from fast_conformer_rnnt_factory import (  # noqa: E402
    build_fast_conformer_rnnt,
    build_streaming_fast_conformer_rnnt,
)


def _load_small_config(name: str) -> DictConfig:
    config = cast(DictConfig, OmegaConf.load(EXPERIMENT_DIR / "config" / f"{name}.yaml"))
    config.frontend.n_mels = 8
    config.model.vocab_size = 16
    config.model.encoder.hidden_size = 8
    config.model.encoder.num_heads = 2
    config.model.encoder.kernel_size = 3
    config.model.encoder.num_blocks = 1
    config.model.encoder.conv_channels = 4
    config.model.prediction_network.hidden_size = 8
    config.model.joint_network.hidden_size = 8
    return config


def test_fast_conformer_factory_builds_only_non_streaming_modules() -> None:
    config = _load_small_config("fast_conformer_rnnt")

    model = build_fast_conformer_rnnt(config, blank_token_id=0)

    assert type(model) is FastConformerRNNT
    assert type(model.encoder) is FastConformer
    assert "min_chunk_size" not in config.model.encoder
    assert "chunk_size" not in config.evaluate
    assert "chunk_size" not in config.infer


def test_streaming_fast_conformer_factory_builds_only_streaming_modules() -> None:
    config = _load_small_config("streaming_fast_conformer_rnnt")

    model = build_streaming_fast_conformer_rnnt(config, blank_token_id=0)

    assert type(model) is StreamingFastConformerRNNT
    assert type(model.encoder) is StreamingFastConformer
    assert config.model.encoder.min_chunk_size > 0
    assert config.evaluate.chunk_size > 0
    assert config.infer.chunk_size > 0
    assert "streaming" not in config.evaluate
    assert "streaming" not in config.infer
