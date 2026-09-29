import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from config.loader import load_config
from engine.trainer import BaseEngine


@pytest.mark.parametrize(
    "losses, expected_epochs",
    [
        ([1.0, 0.9996, 0.9992, 0.9988, 0.9984], 4),
        ([1.0, 0.9996, 0.997, 0.9966, 0.9962, 0.9958], 6),
        ([1.0, 0.999, 0.999, 0.999, 0.999], 5),
        ([1.0, 1.1, 1.2, 1.3, 1.4], 4),
    ],
)
def test_patience_threshold_and_best_checkpoint(tmp_path, losses, expected_epochs):
    engine = BaseEngine.__new__(BaseEngine)
    engine._max_epochs = len(losses)
    engine._patience = 3
    engine._min_delta = 0.001
    engine._lr_scheduler = None
    engine._lrate = 0.001
    engine._save_path = str(tmp_path)
    engine._logger = Mock()
    engine.config = SimpleNamespace(training=SimpleNamespace(evaluate_test=False))
    engine.train_batch = Mock()
    engine.save_model = Mock()
    engine.metric = SimpleNamespace(
        get_valid_loss=lambda: engine.metric.valid_res[0][0],
        get_epoch_msg=lambda *args: "epoch",
        metric_lst=["MAE"],
        loss_name="MAE",
        valid_weights=[1],
    )
    remaining = iter(losses)

    def evaluate(mode):
        assert mode == "val"
        engine.metric.valid_res = [[next(remaining)]]

    engine.evaluate = evaluate
    engine.train()

    assert engine.train_batch.call_count == expected_epochs
    observed = losses[:expected_epochs]
    selection = json.loads((tmp_path / "selection.json").read_text())
    assert selection["validation"]["MAE"] == min(observed)
    assert selection["best_epoch"] == observed.index(min(observed)) + 1
    assert engine.save_model.call_count == sum(
        loss < min(observed[:index], default=float("inf"))
        for index, loss in enumerate(observed)
    )


def test_global_training_defaults():
    config = load_config(model="flow/hl", dataset="dc_60min")
    assert config.training.max_epochs == 400
    assert config.training.patience == 30
    assert config.training.min_delta == 0.001


@pytest.mark.parametrize("value", ["-0.001", ".nan", ".inf"])
def test_invalid_min_delta(value):
    with pytest.raises(ValueError, match="training.min_delta"):
        load_config(
            model="flow/hl", dataset="dc_60min",
            overrides=[f"training.min_delta={value}"],
        )
