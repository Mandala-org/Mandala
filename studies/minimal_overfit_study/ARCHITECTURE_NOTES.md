# Architecture Notes

## Why does TP produce multiple scalar groups?

When you see output like:
```
TP: 64x0e ⊗ 1x0e+1x1o+1x2e+1x3o+1x4e → 64x0e+32x0e+16x0e+8x0e+4x0e+32x1o+16x2e+8x3o+4x4e
```

This is because `FullyConnectedTensorProduct` creates **all possible products**:

- `64x0e ⊗ 1x0e = 64x0e` (scalars ⊗ scalars = scalars)
- `64x0e ⊗ 1x1o = 32x1o` (scalars ⊗ L=1 = L=1, assuming multiplicity halves)
- `64x0e ⊗ 1x2e = 16x2e` (scalars ⊗ L=2 = L=2, multiplicity halves again)
- etc.

The **separate scalar groups** (`64x0e`, `32x0e`, `16x0e`, etc.) come from the tensor product's internal structure - it keeps track of which input irrep produced which output. These are then **combined by the Gate** into a single `64x0e` group in the final output.

**This is correct behavior** - the Gate learns to mix these different scalar contributions appropriately.

## Layer Normalization Count

### Our Architecture (MinimalNetwork with 1 message passing layer):

1. **EdgeEncoder**: 1 norm
   - TP → Gate → **Norm**

2. **MessageBlock**: 2 norms
   - Node update: Linear → Gate → **Norm**
   - Edge update: TP → Gate → **Norm**

**Total: 3 norms per forward pass**

### DeepH-E3 Architecture:

DeepH-E3 uses a similar pattern:
- Initial edge/node encoders have norms
- Each message passing layer has 2 norms (node + edge)
- No doubled norms (each Gate is followed by exactly one Norm)

**Our architecture matches DeepH-E3's norm pattern** ✓

## No Doubled Norms

Looking at the instantiation output:
```
[e3LayerNorm] Irreps: 64x0e+32x1o+16x2e+8x3o+4x4e, affine=True  # EdgeEncoder.norm
[e3LayerNorm] Irreps: 64x0e+32x1o+16x2e+8x3o+4x4e, affine=True  # MessageBlock.edge_norm
[e3LayerNorm] Irreps: 64x0e+32x1o+16x2e+8x3o+4x4e, affine=True  # MessageBlock.node_norm
```

These are **3 different norms in different locations**, not doubled norms. The order is:

1. EdgeEncoder: processes raw edges → outputs normed edge features
2. MessageBlock node update: aggregates edges + node → outputs normed nodes
3. MessageBlock edge update: uses updated nodes + old edges → outputs normed edges

**No sequential norms exist** - each norm is separated by a Gate operation.

## Warning: Missing Irreps in Head

The warnings like:
```
⚠️  WARNING [O-O]: Cannot produce irreps 1e, 2o, 3e, 4e from 64x0e+32x1o+16x2e+8x3o+4x4e
```

This means:
- Your hidden_irreps: `64x0e+32x1o+16x2e+8x3o+4x4e`
- O-O orbital pair needs: `22x0e+30x1o+13x1e+12x2o+25x2e+12x3o+4x3e+4x4e`

**Missing**: `1e` (even L=1), `2o` (odd L=2), `3e` (even L=3), `4e` (even L=4)

**Linear layers can only produce output irreps that match input irreps** - they cannot change parity or create new L values.

### Solution:

Add missing irreps to your hidden_irreps:
```bash
--hidden-irreps "64x0e+32x1o+32x1e+16x2o+16x2e+8x3o+8x3e+4x4o+4x4e"
```

This includes both parities for each L value up to L=4.
