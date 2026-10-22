from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from types import SimpleNamespace


def _load_module():
    path = Path("scripts/optuna_agent.py").resolve()
    spec = importlib.util.spec_from_file_location("optuna_agent_script", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_run_args_from_trial_uses_sampled_and_fixed_values():
    mod = _load_module()

    class FakeTrial:
        def __init__(self):
            self.number = 7

        def suggest_categorical(self, name, choices):
            assert name == "lr"
            return choices[-1]

    study_cfg = {
        "parameters": {
            "data-path": {"value": "/tmp/data"},
            "dataset-kind": {"value": "silicon"},
            "lr": {"values": [0.001, 0.01]},
        }
    }

    args = mod._build_run_args_from_trial(
        FakeTrial(),
        study_cfg,
        checkpoint_dir="checkpoints/optuna",
        wandb_mode="offline",
        study_name="silicon-study",
        agent_label="node01",
    )

    assert args.data_path == "/tmp/data"
    assert args.dataset_kind == "silicon"
    assert args.lr == pytest.approx(0.01)
    assert args.wandb_mode == "offline"
    assert args.run_name == "silicon-study-trial-00007-node01"


def test_build_run_args_from_trial_normalizes_list_values():
    mod = _load_module()

    class FakeTrial:
        number = 0

    study_cfg = {
        "parameters": {
            "data-path": {"value": "/tmp/data"},
            "dataset-kind": {"value": "silicon"},
            "matrix-targets": {"value": "density"},
        }
    }

    args = mod._build_run_args_from_trial(
        FakeTrial(),
        study_cfg,
        checkpoint_dir="checkpoints/optuna",
        wandb_mode="offline",
        study_name="silicon-study",
        agent_label=None,
    )

    assert args.matrix_targets == ["density"]


def test_run_optuna_agent_uses_study_and_calls_training(monkeypatch, tmp_path):
    mod = _load_module()
    study_yaml = tmp_path / "study.yaml"
    study_yaml.write_text(
        """
study_name: silicon-optuna
method: bayes
metric:
  name: val/energy_mae
  goal: minimize
n_trials: 1
parameters:
  data-path:
    value: /tmp/data
  dataset-kind:
    value: silicon
  num-train:
    value: 2
  num-val:
    value: 1
  lr:
    values: [0.001, 0.01]
"""
    )
    captured = {}

    class FakeTrial:
        def __init__(self):
            self.number = 2
            self.attrs = {}

        def suggest_categorical(self, name, choices):
            return choices[0]

        def report(self, value, step):
            captured["report"] = (value, step)

        def should_prune(self):
            return False

        def set_user_attr(self, key, value):
            self.attrs[key] = value

    class FakeStudy:
        def optimize(self, objective, n_trials=None, timeout=None, gc_after_trial=None):
            captured["optimize"] = {
                "n_trials": n_trials,
                "timeout": timeout,
                "gc_after_trial": gc_after_trial,
            }
            trial = FakeTrial()
            captured["objective_value"] = objective(trial)
            captured["user_attrs"] = trial.attrs

    class FakeOptuna:
        class samplers:
            class TPESampler:
                def __init__(self, seed):
                    captured["sampler_seed"] = seed

        class pruners:
            class HyperbandPruner:
                def __init__(self, **kwargs):
                    captured["pruner_kwargs"] = kwargs

        class storages:
            class RDBStorage:
                def __init__(self, **kwargs):
                    captured["storage_kwargs"] = kwargs

        class TrialPruned(Exception):
            pass

        @staticmethod
        def load_study(**kwargs):
            raise RuntimeError("record does not exist")

        @staticmethod
        def create_study(**kwargs):
            captured["create_study"] = kwargs
            return FakeStudy()

    def fake_run_training(
        args, *, parsed_yaml=None, extra_callbacks=None, objective_metric=None
    ):
        captured["train_args"] = args
        captured["parsed_yaml"] = parsed_yaml
        captured["objective_metric"] = objective_metric
        assert extra_callbacks is not None and len(extra_callbacks) == 1
        assert isinstance(extra_callbacks[0], mod.OptunaPruningCallback)
        assert extra_callbacks[0].monitor == objective_metric
        return {"val/energy_mae": 0.123, "val/loss": 0.5}

    monkeypatch.setitem(sys.modules, "optuna", FakeOptuna)
    monkeypatch.setattr(mod, "run_training", fake_run_training)
    monkeypatch.setattr(mod, "_study_init_lock", lambda *args, **kwargs: _null_ctx())

    args = mod.argparse.Namespace(
        study_yaml=str(study_yaml),
        storage="postgresql://user:pass@localhost:5432/mandala",
        study_name=None,
        count=1,
        timeout=None,
        wandb_mode="offline",
        checkpoint_dir="checkpoints/optuna",
        heartbeat_interval=30,
        grace_period=120,
        agent_label=None,
    )

    mod.run_optuna_agent(args)

    assert captured["create_study"]["study_name"] == "silicon-optuna"
    assert captured["create_study"]["direction"] == "minimize"
    assert captured["optimize"]["n_trials"] == 1
    assert captured["objective_metric"] == "val/energy_mae"
    assert captured["train_args"].dataset_kind == "silicon"
    assert captured["train_args"].num_train == 2
    assert captured["objective_value"] == pytest.approx(0.123)
    assert captured["user_attrs"] == {"val/energy_mae": 0.123, "val/loss": 0.5}


def test_get_or_create_study_loads_existing_without_creating(monkeypatch):
    mod = _load_module()
    captured = {}

    class FakeStudy:
        pass

    class FakeOptuna:
        class samplers:
            class TPESampler:
                def __init__(self, seed):
                    pass

        class pruners:
            class HyperbandPruner:
                def __init__(self, **kwargs):
                    pass

        @staticmethod
        def load_study(**kwargs):
            captured["load_study"] = kwargs
            return FakeStudy()

        @staticmethod
        def create_study(**kwargs):
            pytest.fail("An existing study must not be recreated")

    study_cfg = {
        "study_name": "demo",
        "metric": {"name": "val/loss", "goal": "minimize"},
        "parameters": {},
    }

    monkeypatch.setattr(mod, "_study_init_lock", lambda *args, **kwargs: _null_ctx())

    study = mod._get_or_create_study(
        FakeOptuna,
        "demo",
        storage=object(),
        study_cfg=study_cfg,
        study_yaml_path="study.yaml",
        storage_url="postgresql://example",
    )

    assert isinstance(study, FakeStudy)
    assert captured["load_study"]["study_name"] == "demo"


def test_get_or_create_study_creates_when_missing(monkeypatch):
    mod = _load_module()
    captured = {}

    class FakeStudy:
        pass

    class FakeOptuna:
        class samplers:
            class TPESampler:
                def __init__(self, seed):
                    pass

        class pruners:
            class HyperbandPruner:
                def __init__(self, **kwargs):
                    pass

        @staticmethod
        def load_study(**kwargs):
            raise RuntimeError("record does not exist")

        @staticmethod
        def create_study(**kwargs):
            captured["create_study"] = kwargs
            return FakeStudy()

    study_cfg = {
        "study_name": "demo",
        "metric": {"name": "val/loss", "goal": "minimize"},
        "parameters": {},
    }

    monkeypatch.setattr(mod, "_study_init_lock", lambda *args, **kwargs: _null_ctx())

    study = mod._get_or_create_study(
        FakeOptuna,
        "demo",
        storage=object(),
        study_cfg=study_cfg,
        study_yaml_path="study.yaml",
        storage_url="postgresql://example",
    )

    assert isinstance(study, FakeStudy)
    assert captured["create_study"]["study_name"] == "demo"
    assert captured["create_study"]["load_if_exists"] is True
    assert captured["create_study"]["direction"] == "minimize"


@pytest.mark.parametrize("metric", [0.125, torch.tensor(0.125)])
@pytest.mark.parametrize("prune", [False, True])
def test_pruning_callback_reports_the_monitored_epoch_and_honors_decision(
    metric, prune
):
    import optuna

    mod = _load_module()
    reports = []
    trial = SimpleNamespace(
        report=lambda value, step: reports.append((value, step)),
        should_prune=lambda: prune,
    )
    trainer = SimpleNamespace(
        sanity_checking=False,
        current_epoch=7,
        callback_metrics={"val/loss": metric, "train/loss": 99.0},
    )
    callback = mod.OptunaPruningCallback(trial, "val/loss")
    if prune:
        with pytest.raises(optuna.TrialPruned, match="epoch 7"):
            callback.on_validation_epoch_end(trainer, None)
    else:
        callback.on_validation_epoch_end(trainer, None)
    assert reports == [(0.125, 7)]


@pytest.mark.parametrize("case", ["sanity", "missing", "nan", "inf"])
def test_pruning_callback_ignores_unusable_validation_metrics(case):
    mod = _load_module()

    def unexpected(*args, **kwargs):
        pytest.fail("A sanity/missing/nonfinite metric must not reach the trial")

    trial = SimpleNamespace(report=unexpected, should_prune=unexpected)
    trainer = SimpleNamespace(
        sanity_checking=case == "sanity",
        current_epoch=0,
        callback_metrics={
            "val/loss": {
                "sanity": 1.0,
                "missing": None,
                "nan": float("nan"),
                "inf": float("inf"),
            }[case]
        },
    )
    mod.OptunaPruningCallback(trial, "val/loss").on_validation_epoch_end(trainer, None)


def test_study_connection_failure_is_not_treated_as_a_missing_study(monkeypatch):
    mod = _load_module()
    import optuna

    def unavailable(**kwargs):
        raise ConnectionError("database unreachable")

    monkeypatch.setattr(optuna, "load_study", unavailable)
    monkeypatch.setattr(
        optuna,
        "create_study",
        lambda **kwargs: pytest.fail("Must not create on connection failure"),
    )
    monkeypatch.setattr(mod, "_study_init_lock", lambda *args: _null_ctx())
    with pytest.raises(ConnectionError, match="database unreachable"):
        mod._get_or_create_study(
            optuna,
            "demo",
            object(),
            {"metric": {"goal": "minimize"}},
            study_yaml_path="study.yaml",
            storage_url="postgresql://example",
        )


class _null_ctx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False
