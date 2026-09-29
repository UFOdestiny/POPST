"""Regression checks for OD axes, causal history and baseline-specific paths."""

from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch

from config.loader import load_config, resolved_copy
from engine.runner import load_recipe


NEURAL = ["agcrn", "astgcn", "gwnet", "stgcn", "stgode", "stzinb", "stpro", "od_ced", "hl", "odmixer"]


@pytest.fixture
def history(tmp_path):
    torch.set_num_threads(2)
    folder = tmp_path / "v"
    folder.mkdir()
    rng = np.random.default_rng(3)
    data = rng.random((144, 4, 4, 2), dtype=np.float32)
    np.savez(folder / "his.npz", data=data)
    np.save(folder / "idx_train.npy", np.arange(11, 93))
    np.save(tmp_path / "adj.npy", np.eye(4, dtype=np.float32) + .2)
    (folder / "meta.json").write_text(json.dumps({"scaler_params": {"data_min": [0, 0], "data_max": [3, 3], "use_log1p": True}}))
    return tmp_path


@pytest.mark.parametrize("name", NEURAL)
def test_neural_multihorizon_and_channel_order(name, history):
    cfg = load_config(model=f"od/{name}", dataset="dc_od_60min")
    cfg = resolved_copy(cfg, {"data": {"node_num": 4, "input_dim": 4, "output_dim": 4, "seq_len": 12,
                                      "horizon": 3, "version": "v"}, "runtime": {"device": "cpu"}})
    recipe = load_recipe(cfg.model.id)
    ctx = recipe.setup(cfg, str(history), str(history / "adj.npy"), 4, torch.device("cpu"), None) if recipe.setup else {}
    model = recipe.build_model(cfg, 4, **ctx).eval()
    # AGCRN's source weight pools are initialized by its recipe engine.
    if recipe.init_weights:
        for p in model.parameters():
            torch.nn.init.xavier_uniform_(p) if p.ndim > 1 else torch.nn.init.uniform_(p)
    x = torch.rand(2, 12, 4, 4, 2)
    inp = {"od": x, "prev_od": x.flip(1)} if name == "odmixer" else x
    with torch.no_grad():
        output = model(inp)
        point = output[0] if isinstance(output, tuple) else output
        assert point.shape == (2, 3, 4, 4, 2)
        assert torch.isfinite(point).all()
        for channel in range(2):
            single = {k: v[..., channel:channel+1] for k, v in inp.items()} if isinstance(inp, dict) else inp[..., channel:channel+1]
            actual = model(single)
            actual = actual[0] if isinstance(actual, tuple) else actual
            torch.testing.assert_close(point[..., channel:channel+1], actual, rtol=2e-4, atol=2e-5)
        # A sample must not depend on other samples in the evaluation batch.
        first = {k: v[:1] for k, v in inp.items()} if isinstance(inp, dict) else inp[:1]
        actual = model(first)
        actual = actual[0] if isinstance(actual, tuple) else actual
        torch.testing.assert_close(point[:1], actual, rtol=2e-4, atol=2e-5)


def test_odmixer_uses_past_branch_and_never_future_auxiliary():
    from models.od.odmixer import ODMixerDataset, ODMixer
    raw = np.arange(100, dtype=np.float32).reshape(100, 1, 1, 1)
    ds = ODMixerDataset(raw, [40], 12, 3, 24)
    x, y = ds[0]
    assert x["od"][-1].item() == 40 and y[0].item() == 41
    assert x["prev_od"][-1].item() == 16 and x["prev_y_od"][-1].item() == 19
    with pytest.raises(ValueError, match="observed"):
        ODMixerDataset(raw, [40], 12, 25, 24)
    model = ODMixer(12, 2, 8, 2, 2, 0., 2, 3, 2).eval()
    current = torch.rand(1, 12, 2, 2, 1, requires_grad=True)
    past = torch.rand_like(current, requires_grad=True)
    a, b = model({"od": current, "prev_od": past}, return_aux=True)
    (a.square().mean() + b.square().mean()).backward()
    assert past.grad.abs().sum() > 0


def test_simple_references_same_origins_and_causal():
    from models.od.ha import HA
    from models.od.seasonal import SeasonalNaive
    from models.od.persistence import Persistence
    from models.od.zero import Zero
    raw = np.arange(60, dtype=np.float32).reshape(60, 1, 1, 1)
    args = dict(node_num=1, input_dim=1, output_dim=1, seq_len=12, horizon=6)
    idx = np.array([30, 31])
    for model in [HA(step=3, **args), SeasonalNaive(period=4, **args), Persistence(**args), Zero(**args)]:
        pred = model.forecast_origins(raw, 20, idx, 6)
        assert pred.shape == (2, 6, 1, 1, 1)
        future = raw.copy()
        future[31:] += 1000
        np.testing.assert_array_equal(pred[0], model.forecast_origins(future, 20, idx, 6)[0])
    assert HA(step=3, **args).forecast_origins(raw, 20, idx, 6)[0, 0].item() == 29


def test_nonfinite_forecast_cannot_score_as_zero_error():
    from engine.metrics import masked_mae
    assert torch.isnan(masked_mae(torch.tensor([float("nan")]), torch.tensor([1.]), float("nan")))
    assert masked_mae(torch.tensor([float("nan"), 2.]), torch.tensor([float("nan"), 1.]), float("nan")).item() == 1


def test_stgode_graph_ignores_future(history):
    from models.od.stgode import _construct_se_matrix
    cfg = resolved_copy(load_config(model="od/stgode", dataset="dc_od_60min"), {"data": {"version": "v", "horizon": 3}})
    before = _construct_se_matrix(history, cfg)
    with np.load(history / "v/his.npz") as archive:
        data = archive["data"].copy()
    data[96:] = 100
    np.savez(history / "v/his.npz", data=data)
    np.testing.assert_array_equal(before, _construct_se_matrix(history, cfg))


def test_od_ced_coarsening_ignores_future_and_adjacency(history):
    from models.od.od_ced import setup
    cfg = resolved_copy(load_config(model="od/od_ced", dataset="dc_od_60min"), {
        "data": {"version": "v", "horizon": 3}})
    before = setup(cfg, str(history), "unused", 4, "cpu", None)
    with np.load(history / "v/his.npz") as archive:
        data = archive["data"].copy()
    data[96:] = 100
    np.savez(history / "v/his.npz", data=data)
    after = setup(cfg, str(history), "nonexistent adjacency", 4, "cpu", None)
    for name in ("assignment", "fine_mask", "coarse_mask"):
        np.testing.assert_array_equal(before[name], after[name])


def test_od_ced_aggregates_counts_and_trains_both_heads(history):
    from models.od.od_ced import ODCED, masked_mse, semantic_coarsening
    from data.preprocessing import MinMaxScaler
    assignment, fine_mask, coarse_mask = semantic_coarsening(np.array([
        [0, 5, 2, 0], [3, 0, 0, 1], [2, 0, 0, 4], [0, 1, 4, 0]], float), .5)
    assert assignment.shape == (4, 2)
    scaler = MinMaxScaler(use_log1p=True).fit(np.array([[0, 0], [10, 100]], np.float32), per_channel=True)
    model = ODCED(4, 4, 4, 12, 3, assignment, fine_mask, coarse_mask, scaler)
    counts = torch.arange(1, 1 + 2 * 12 * 4 * 4 * 2, dtype=torch.float32).reshape(2, 12, 4, 4, 2) / 20
    x = model.transform(counts)
    aggregated = model.inverse(model.aggregate(x))
    torch.testing.assert_close(aggregated.sum((2, 3)), counts.sum((2, 3)))
    expected = torch.zeros_like(aggregated)
    for origin in range(4):
        for destination in range(4):
            expected[:, :, assignment[origin].argmax(), assignment[destination].argmax()] += counts[:, :, origin, destination]
    torch.testing.assert_close(aggregated, expected)
    point, coarse = model(x, return_aux=True)
    loss = masked_mse(point, torch.zeros_like(point), model.fine_mask[None, None, :, :, None])
    loss += masked_mse(coarse, torch.zeros_like(coarse), model.coarse_mask[None, None, :, :, None])
    loss.backward()
    assert model.outconv[-1].weight.grad.abs().sum() > 0
    assert model.outconv_enc[-1].weight.grad.abs().sum() > 0
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert masked_mse(point, point, torch.zeros_like(point, dtype=torch.bool)).item() == 0


def test_od_ced_attention_preserves_allowed_zero_logits():
    from models.od.od_ced import CAttention
    attention = CAttention(4, 2)
    with torch.no_grad():
        attention.q.weight.zero_()
        attention.k.weight.zero_()
        attention.v.weight.copy_(torch.eye(4))
        attention.proj.weight.copy_(torch.eye(4))
        attention.proj.bias.zero_()
    values = torch.tensor([[[1., 2, 3, 4], [100., 200, 300, 400]]])
    allowed = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.bool)
    actual = attention(torch.zeros_like(values), values, values, allowed)
    torch.testing.assert_close(actual, values)


def test_multihorizon_split_has_no_target_overlap():
    from data.preprocessing import _split_by_ratio
    train, val, test, _ = _split_by_ratio(np.zeros((100, 1)), SimpleNamespace(seq_length_x=12, seq_length_y=4))
    assert train[-1] + 4 < val[0] + 1
    assert val[-1] + 4 < test[0] + 1


@pytest.mark.parametrize("log1p", [False, True])
def test_train_constant_channel_preserves_future_counts(log1p):
    from data.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(use_log1p=log1p).fit(np.zeros((50, 2)), per_channel=True)
    later = np.array([[0., 10.], [3., 20.]], dtype=np.float32)
    for value in (later, torch.from_numpy(later)):
        normalized = scaler.transform(value)
        restored = scaler.inverse_transform(normalized)
        np.testing.assert_allclose(np.asarray(restored), later, rtol=1e-6, atol=1e-6)


def test_preparation_fits_only_training_observations(tmp_path):
    from data.prepare import prepare_array
    raw = np.zeros((1, 1, 100), dtype=np.float32)
    raw[..., 90:] = 100
    source = tmp_path / "raw.npy"
    np.save(source, raw)
    output = prepare_array(source, "fixture", "v", tmp_path / "output", per_channel=True)
    meta = json.loads((output / "meta.json").read_text())
    assert meta["scaler_params"]["data_max"] == [0.]
    with np.load(output / "his.npz") as saved:
        assert saved["data"][-1].item() == 100


def test_aci_waits_for_each_horizon_target():
    from engine.calibration.od_aci import OD_ACI_Engine
    engine = OD_ACI_Engine.__new__(OD_ACI_Engine)
    engine.alpha, engine.gamma, engine.cqr_mode = .1, .05, "horizon"
    engine._aci_reference = [np.arange(100)] * 3
    engine._aci_alpha = np.full(3, .1)
    engine._device = torch.device("cpu")
    engine.model = SimpleNamespace(horizon=3)
    engine._dataloader = {"test_loader": SimpleNamespace(dataset=SimpleNamespace(indices=np.arange(10, 14)))}
    pred = torch.zeros(4, 3, 1, 1, 1)
    engine._collect = lambda mode: (pred, torch.full_like(pred, 1000.))
    engine.metric = SimpleNamespace(compute_one_batch=lambda *a, **kw: None)
    snapshots = []
    issue = engine._interval_at_origin

    def record(point):
        snapshots.append(engine._aci_alpha.copy())
        return issue(point)

    engine._interval_at_origin = record
    engine.evaluate("test", train_test=True)
    np.testing.assert_allclose(snapshots[0], [.1, .1, .1])
    np.testing.assert_allclose(snapshots[1], [.055, .1, .1])
    np.testing.assert_allclose(snapshots[2], [.01, .055, .1])
    np.testing.assert_allclose(snapshots[3], [1e-6, .01, .055])


@pytest.mark.parametrize("name", ["arima", "sarima", "var"])
def test_statistical_forecasts_only_use_observed_history(name):
    cfg = load_config(model=f"od/{name}", dataset="dc_od_60min")
    params = {"order": [1, 0, 0], "n_threads": 1}
    if name == "sarima":
        params["seasonal_order"] = [1, 0, 0, 4]
    elif name == "var":
        params = {"k": 2, "lags": 2}
    cfg = resolved_copy(cfg, {"model": {"params": params}, "data": {
        "node_num": 2, "input_dim": 2, "output_dim": 2, "horizon": 3}})
    model = load_recipe(cfg.model.id).build_model(cfg, 2)
    raw = np.random.default_rng(2026).normal(5, 1, (100, 2, 2, 1)).astype(np.float32)
    origins = np.array([75, 80])
    pred = model.forecast_origins(raw, 60, origins, 3)
    assert pred.shape == (2, 3, 2, 2, 1) and np.isfinite(pred).all()
    changed = raw.copy()
    changed[76:] += 1000
    np.testing.assert_allclose(pred[0], model.forecast_origins(changed, 60, origins, 3)[0])
