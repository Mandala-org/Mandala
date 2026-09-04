import importlib.util
from pathlib import Path

import pytest
import torch


def _load_precompute_module():
    path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "pair_descriptors"
        / "precompute_d1_sio2.py"
    )
    spec = importlib.util.spec_from_file_location("precompute_d1_sio2", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_worker_initializer_is_idempotent():
    module = _load_precompute_module()
    module._initialize_worker()
    module._initialize_worker()
    assert torch.get_num_threads() == 1
