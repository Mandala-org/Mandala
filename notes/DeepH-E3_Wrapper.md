### Updated Implementation Plan for DeepH-E3 Wrapper

This document provides a detailed plan for wrapping the DeepH-E3 model and integrating it into the Mandala framework, based on a corrected understanding of the DeepH-E3 data and basis convention pipeline.

---

### DeepH-E3 Basis Convention Pipeline (Corrected)

The key finding is that the loss comparison in DeepH-E3 is performed in the **OpenMX basis**.

1.  **Data Preparation (`deephe3/data.py: AijData.set_mask`)**
    *   With the default setting `convert_net_out = False`, raw Hamiltonian blocks are loaded from HDF5 files in the **OpenMX basis**.
    *   These blocks are flattened and concatenated into the `data.label` tensor, which remains in the **OpenMX basis**.

2.  **Network Prediction (`deephe3/model.py: Net.forward`)**
    *   The network outputs `edge_fea`, a tensor of predicted irrep coefficients, which are in the **e3nn/wiki basis**.

3.  **Prediction Transformation & Loss (`deephe3/kernel.py: DeepHE3Kernel.get_loss`)**
    *   The code calls `H_pred = construct_kernel.get_H(output_edge)`.
    *   Inside `get_H` (`deephe3/e3modules.py`), the predicted irreps (e3nn/wiki basis) are first used to construct matrix blocks in the **e3nn/wiki basis**.
    *   Then, `get_H` explicitly converts these predicted blocks from the **e3nn/wiki basis back to the OpenMX basis** using `rotate_kernel.wiki2openmx_H`.
    *   The final loss is computed between `H_pred` (predicted blocks, now in **OpenMX basis**) and `batch.label` (ground truth blocks, also in **OpenMX basis**).

---

### Implementation Plan

#### **Part 1: Data Convention Alignment in Mandala**

To align with DeepH-E3, Mandala's data loader must be configured to provide target matrices in the OpenMX basis.

1.  **Modify `data/factory.py`**: Add a `convention="e3nn"` argument to the `DatasetFactory` constructor.
2.  **Modify `data/gnn_dataset.py`**: The `E3GNNDataset` class will accept the `convention` argument and pass it to the `Snapshot.from_openmx` method during snapshot processing.
3.  **Modify Training Script (`scripts/train_silicon_deephe3.py`)**: When instantiating `DatasetFactory`, explicitly pass `convention="openmx"`. Also, ensure the script is configured to only use the Hamiltonian as a target by setting `cfg.matrix_targets = ["hamiltonian"]`.

#### **Part 2: Making DeepH-E3 Importable**

To use the `deephe3` package, its parent directory will be added to the system path at the top of `src/net/e3gnn.py`.

```python
import sys
from pathlib import Path

deeph_path = Path(__file__).resolve().parents[2] / "external" / "DeepH-E3"
if str(deeph_path) not in sys.path:
    sys.path.append(str(deeph_path))

from deephe3.model import Net as DeepHE3Net
from deephe3.e3modules import e3TensorDecomp, Rotate
from deephe3.utils import MaskMSELoss
```

#### **Part 3: Implementing the `DeepHE3` Wrapper Class**

The wrapper will be implemented in `src/net/e3gnn.py`.

1.  **`__init__(self, cfg: Config, mapper: BlockIrrepMapper)`**:
    *   The `mapper` will be used only to extract orbital configurations and species information needed to initialize DeepH-E3's modules.
    *   Hyperparameters for `DeepHE3Net` will be hardcoded based on the defaults in `train_default.ini`.
    *   It will instantiate `DeepHE3Net`, `e3TensorDecomp`, and `Rotate`.

2.  **`_prepare_deeph_input(self, x: Dict[str, Any]) -> torch_geometric.data.Batch`**:
    *   This method will convert Mandala's `x` dictionary to a `torch_geometric.data.Batch` object.
    *   It will correctly calculate displacement vectors for `edge_attr` respecting PBC, using the same logic as DeepH-E3.

3.  **`forward(self, x: Dict[str, Any]) -> torch.Tensor`**:
    *   This method will call `_prepare_deeph_input` and then `self.deephe3.forward()`, returning the raw `edge_fea` tensor (predicted irreps in e3nn/wiki basis).

4.  **`_reconstruct_block_matrix(self, predicted_irreps, x) -> BlockMatrix`**:
    *   This new helper method will take the predicted irreps and reconstruct a Mandala `BlockMatrix`.
    *   It will call `self.construct_kernel.get_H(predicted_irreps)` to get the flattened matrix prediction in the **OpenMX basis**.
    *   It will then use logic adapted from DeepH-E3's `update_hopping` function to reshape the flat vector back into a dictionary of matrix blocks.
    *   Finally, it will construct and return a `BlockMatrix` object from these blocks.

5.  **`_shared_step(self, batch, batch_idx, stage)`**:
    *   It will call `forward` to get predictions and `_reconstruct_block_matrix` to get a predicted `BlockMatrix` in the **OpenMX basis**.
    *   The ground truth `y['hamiltonian']` will also be a `BlockMatrix` in the **OpenMX basis** (due to the changes in Part 1).
    *   The loss will be calculated by comparing the blocks of the predicted and ground truth `BlockMatrix` objects using Mandala's existing MSE logic.
