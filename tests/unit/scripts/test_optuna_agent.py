from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


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
            objective(FakeTrial())

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
        return {"val/energy_mae": 0.123, "val/loss": 0.5}

    monkeypatch.setitem(sys.modules, "optuna", FakeOptuna)
    monkeypatch.setattr(mod, "run_training", fake_run_training)

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
