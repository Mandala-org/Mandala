# E3GNN Architectural Variants Report

This document provides an overview of the possible architectural variants of the `E3GNN` framework implemented in this project. The flexibility of the architecture is primarily controlled by the `Config` dataclass defined in `net/common.py`.

## 1. Introduction

The core of the network is an E(3)-equivariant graph neural network that operates on atomic structures. It consists of three main stages:
1.  **Encoding:** Node and edge features are encoded into equivariant representations.
2.  **Message Passing:** A stack of message passing layers iteratively updates node and edge features.
3.  **Decoding (Heads):** A set of deep, equivariant heads reads out the final features to predict the Hamiltonian, Overlap, and Density matrices.

The framework allows for significant variation within the message passing and decoding stages, which are detailed below.

## 2. Message Passing Layers

The message passing is performed by the `MessageBlock` module (`net/layers.py`), which is composed of an `EdgeUpdateBlock` and a `NodeUpdateBlock`. The behavior of these blocks can be configured to create different network variants.

### 2.1. Edge Update Block

The `EdgeUpdateBlock` updates the feature vector $e_{ij}$ for each edge $(i \to j)$ based on the features of the source node $h_i$ and destination node $h_j$.

#### 2.1.1. Node Feature Combination (`edge_update_node_combine`)

This setting determines how the features from the source and destination nodes are combined to form an initial message.

-   **`sum`**: The node features are summed. This is a symmetric operation.
    $$
    \mathbf{m}_{ij}^{(0)} = \text{Linear}(h_i + h_j)
    $$
-   **`concat`**: The node features are concatenated. This is an asymmetric operation, allowing the model to distinguish between source and destination.
    $$
    \mathbf{m}_{ij}^{(0)} = \text{Linear}(\text{concat}(h_i, h_j))
    $$

#### 2.1.2. Edge Feature Update (`edge_update`)

This setting controls how the message $\mathbf{m}_{ij}^{(0)}$ is combined with the existing edge features $e_{ij}$ from the previous layer.

-   **`tensor_product`**: The new edge features are formed by the tensor product of the node-derived message and the old edge features.
    $$e'_{ij} = \mathbf{m}_{ij}^{(0)} \otimes e_{ij}
    $$
-   **`concat`**: The features are concatenated and passed through a linear layer.
    $$e'_{ij} = \text{Linear}(\text{concat}(\mathbf{m}_{ij}^{(0)}, e_{ij}))
    $$
-   **`replace`**: The old edge features are completely replaced by the node-derived message.
    $$e'_{ij} = \text{Linear}(\mathbf{m}_{ij}^{(0)})
    $$

#### 2.1.3. Residual Connection (`edge_update_residual`)

If `True`, a residual connection is added after the non-linearity, adding the original edge features to the updated ones.

$$e_{ij}^{(\text{new})} = \text{Activation}(e'_{ij}) + e_{ij}
$$

### 2.2. Node Update Block

The `NodeUpdateBlock` updates the feature vector $h_i$ for each node by aggregating messages from all incoming edges.

#### 2.2.1. Message Aggregation (`node_update_message_agg`)

This setting determines how incoming edge features are aggregated at each node.

-   **`sum`**: A simple sum of all incoming edge features.
    $$
    \mathbf{m}_i = \sum_{j \in \mathcal{N}(i)} e_{ji}
    $$
-   **`attention`**: A simple attention mechanism is used. The edge features are split into queries, keys, and values ($q, k, v$). Attention weights are computed from the dot product of queries and keys, and a weighted sum of the values is performed.
    $$
    \alpha_{ji} = \text{softmax}_j(q_{ji} \cdot k_{ji})
    $$
    $$
    \mathbf{m}_i = \sum_{j \in \mathcal{N}(i)} \alpha_{ji} v_{ji}
    $$

#### 2.2.2. Node Feature Update (`node_update`)

This setting controls how the aggregated message $\mathbf{m}_i$ is combined with the existing node features $h_i$.

-   **`tensor_product`**: The new node features are the tensor product of the old features and the aggregated message.
    $$h'_{i} = h_i \otimes \mathbf{m}_i
    $$
-   **`concat`**: The features are concatenated and passed through a linear layer.
    $$h'_{i} = \text{Linear}(\text{concat}(h_i, \mathbf{m}_i))
    $$
-   **`replace`**: The old node features are replaced by the aggregated message.
    $$h'_{i} = \text{Linear}(\mathbf{m}_i)
    $$
-   **`sum`**: The features are summed.
    $$h'_{i} = h_i + \mathbf{m}_i
    $$

#### 2.2.3. Residual Connection (`node_update_residual`)

If `True`, a residual connection is added after the non-linearity.

$$h_{i}^{(\text{new})} = \text{Activation}(h'_{i}) + h_{i}
$$

## 3. Equivariant Non-linearities (`nonlin_kind`)

The framework supports several types of equivariant non-linearities, which are applied after linear and tensor product operations in the update blocks and heads.

-   **`normact`**: `e3nn.nn.NormActivation`. This computes the norm of each irreducible representation, passes the norms through a scalar activation function, and rescales the original vector.
-   **`s2act`**: `e3nn.nn.S2Activation`. This applies a non-linearity by interpreting the features as signals on the sphere $S^2$.
-   **`gate_scalars_mlp`**: A custom gate activation where the gate for non-scalar irreps is controlled by an MLP applied to the scalar (`0e`) components of the feature vector.
-   **`gate_magnitudes`**: A custom gate activation where the gate for non-scalar irreps is computed from their own magnitudes.

The specific scalar activations used within these modules are also configurable via `activation_scalar` and `activation_gate`.

## 4. Output Heads (`DeepHead`)

The `DeepHead` module decodes the final edge features into the target matrices. Its architecture can be varied.

-   **`neck_depth`**: Controls the number of layers in the shared "trunk" MLP that processes the final edge features before the final projection.
-   **`head_depth`**: Controls the number of layers in the per-pair MLPs that come after the trunk.
-   **`head_use_mlp_log_scale`**: If `True`, an additional small MLP is used to predict a logarithmic scaling factor that is applied to the final output vectors. This can help control the magnitude of the initial predictions.

By combining these options, a wide variety of E(3)-equivariant GNN architectures can be instantiated and tested within this single framework.
