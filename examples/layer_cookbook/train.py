"""Train a recipe that uses the cookbook layers with the standard trainer.

Registering transformations is a side effect of importing their module, so a
custom-layer training run is the ordinary ``scripts.train`` command preceded
by one import. All command-line options of ``scripts.train`` are accepted::

    PYTHONPATH=src:. python examples/layer_cookbook/train.py \
        --config examples/layer_cookbook/cookbook.yaml \
        --set 'data.splits.train=[data/small/Si]' \
        --set 'data.splits.val=[data/small/Si_perturbed_scale1_078]' \
        --output runs/cookbook_silicon --accelerator cpu
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layers  # noqa: E402,F401  (registers the cookbook transformations)

from scripts.train import main  # noqa: E402

if __name__ == "__main__":
    main()
