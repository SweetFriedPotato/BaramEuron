from pathlib import Path

import numpy as np
import torch

from experiments.exp04_raw_grid_spatiotemporal.src.models import build_model
from experiments.exp04_raw_grid_spatiotemporal.src.raw_preprocessing import FoldRawPreprocessor
from experiments.exp04_raw_grid_spatiotemporal.src.run_experiment import load_variant_config
from experiments.exp04_raw_grid_spatiotemporal.src.source_fusion import LeadTimeGatedFusion
from experiments.exp04_raw_grid_spatiotemporal.src.spatial_encoder import GroupQuerySpatialAttention
from experiments.exp04_raw_grid_spatiotemporal.src.trainer import RawGridDataset, RawModelInputs


ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / "experiments/exp04_raw_grid_spatiotemporal/configs"


def test_preprocessing_uses_fit_values_and_handles_future_missing():
    train_ldaps = np.arange(2 * 24 * 2 * 3, dtype=np.float32).reshape(2, 24, 2, 3)
    train_gfs = train_ldaps[:, :, :1, :].copy()
    processor = FoldRawPreprocessor().fit(train_ldaps, train_gfs)
    medians = processor.states["ldaps"].median.copy()
    future_ldaps = np.full((1, 24, 2, 3), np.nan, dtype=np.float32)
    future_gfs = np.full((1, 24, 1, 3), 1e9, dtype=np.float32)
    left, right = processor.transform(future_ldaps, future_gfs)
    assert np.array_equal(processor.states["ldaps"].median, medians)
    assert np.isfinite(left).all() and np.isfinite(right).all()
    assert np.max(right) < 10.0


def test_attention_shape_sum_and_positive_distance_bias():
    module = GroupQuerySpatialAttention(10, 11, token_dim=64, heads=4, use_geo=True).eval()
    dynamic = torch.randn(2, 24, 16, 10)
    static = torch.rand(3, 16, 11)
    output, attention, _ = module(dynamic, static)
    assert output.shape == (2, 24, 3, 64)
    assert attention.shape == (2, 24, 3, 16)
    assert torch.allclose(attention.sum(dim=-1), torch.ones(2, 24, 3), atol=1e-5)
    assert torch.all(module.beta_distance > 0)
    assert torch.all(module.beta_height > 0)


def test_source_gate_is_bounded_and_group_shaped():
    fusion = LeadTimeGatedFusion(64)
    left, right = torch.randn(2, 24, 3, 64), torch.randn(2, 24, 3, 64)
    output, gate = fusion(left, right, torch.rand(2, 24, 3), torch.rand(2, 24, 3))
    assert output.shape == (2, 24, 3, 64)
    assert gate.shape == (2, 24, 3, 1)
    assert torch.all((gate >= 0) & (gate <= 1))


def test_hybrid_forward_backward_and_attention_contract():
    config = load_variant_config(CONFIGS / "raw_hybrid_gated.yaml")
    model = build_model(
        config, 16, 26, np.random.randn(3, 16, 11).astype("float32"),
        np.random.randn(3, 9, 11).astype("float32"), 5, (4, 4, 4),
    )
    power, auxiliary, diagnostics = model(
        torch.randn(2, 24, 16, 16), torch.randn(2, 24, 9, 26),
        torch.randn(2, 24, 5), torch.randn(2, 24, 3, 4),
    )
    (power.mean() + auxiliary.mean()).backward()
    assert power.shape == auxiliary.shape == (2, 24, 3)
    assert diagnostics["ldaps_attention"].shape == (2, 24, 3, 16)
    assert diagnostics["gfs_attention"].shape == (2, 24, 3, 9)
    assert diagnostics["source_gate"].shape == (2, 24, 3, 1)


def test_scada_is_auxiliary_target_not_dataset_input():
    inputs = RawModelInputs(
        np.zeros((2, 24, 16, 10), dtype=np.float32),
        np.zeros((2, 24, 9, 19), dtype=np.float32),
        np.zeros((2, 24, 0), dtype=np.float32),
        np.zeros((2, 24, 3, 0), dtype=np.float32),
    )
    dataset = RawGridDataset(inputs, auxiliary=np.ones((2, 24, 3), dtype=np.float32),
                             auxiliary_mask=np.ones((2, 24, 3), dtype=bool))
    assert RawGridDataset.input_names == (
        "ldaps_dynamic", "gfs_dynamic", "engineered_common", "engineered_group"
    )
    assert len(dataset[0]) == 8
