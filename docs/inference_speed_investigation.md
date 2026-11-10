# Why the in-MANDALA reference presets look slower at inference

**11 November 2026.** Follow-up to the inference-latency benchmark
(`researchops/runs/mandala_inference_speed_v1`, attempt 4, jobs 836943/836944),
which reported the in-MANDALA DeepH-E3 and MACE-H presets 2.3–3.4× slower than
the native implementations in "forward" and 1.1–1.3× slower end to end.

## Short answer

The network arithmetic is essentially the same. The gap comes from two
framework functions that MANDALA's `E3GNN.forward` runs on every call:

1. **`IrrepsBlockData.from_packed`** (`src/data/block_matrix.py`), which wraps
   the packed output into per-species-pair blocks. It loops over every edge in
   Python and calls `int()` / `.tolist()` on CUDA tensor elements. Each of those
   is a GPU→CPU synchronisation. Cost: ~150 ms (graphene), ~450 ms (MoS₂).
   **The native path pays exactly the same cost**, but in its decode step,
   outside the benchmark's forward timer. Most of the 2.3–3.4× "forward" gap is
   this accounting difference, not extra work.
2. **`build_trace_alignment_from_pair_edges`** (`src/core/sparse_math.py`),
   called from `complete_graph_metadata` → `build_prediction_edge_metadata`.
   It builds reverse-edge alignment for trace and spectral losses, again with a
   per-edge `.tolist()` on CUDA rows. Cost: ~124 ms (graphene), ~157 ms (MoS₂).
   Inference never uses its output, and the native path never computes it.
   **This is the genuine end-to-end overhead.**

Together they cause 23.5k (graphene) and 63.6k (MoS₂) device-to-host syncs per
in-MANDALA forward, against 19 for native DeepH-E3. Vectorised replacements
that sync once per call give **identical output** (verified on the arguments
captured from every real call). With both replacements, the remaining
end-to-end difference is 0–6%.

## Measurements

H100, the benchmark's own configs, structures, seed, float32, eager e3nn and
frozen framework (commit 3ec49ae + attempt-4 patch), untrained weights. Jobs
838693 (graphene) and 838694 (MoS₂). Script: `scripts/profile_reference_inference.py`.

### Reproduction of the benchmark (uninstrumented medians, ms)

| Variant | Graphene fwd | Graphene e2e | MoS₂ fwd | MoS₂ e2e | Benchmark (fwd / e2e) |
|---|---:|---:|---:|---:|---|
| DeepH | 5.8 | 1049 | 6.6 | 2029 | 5.8/1050, 6.6/2036 |
| DeepH in MANDALA | 285 | 1178 | 591 | 2177 | 286/1183, 599/2258 |
| DeepH-E3 | 234 | 410 | 245 | 747 | 233/411, 251/762 |
| DeepH-E3 in MANDALA | 526 | 546 | 846 | 917 | 527/549, 862/929 |
| MACE-H | 234 | 407 | 253 | 746 | 236/416, 261/745 |
| MACE-H in MANDALA | 532 | 552 | 862 | 929 | 533/553, 865/936 |
| MANDALA | 270 | 554 | 352 | 725 | 267/553, 331/700 |

### Where the time goes (CUDA-synchronised inclusive timers, ms per structure)

| | Network | `from_packed` | Trace alignment | Other preamble | Decode |
|---|---:|---:|---:|---:|---:|
| Graphene DeepH-E3 native | 234 | (151, in decode) | — | — | 163 |
| Graphene DeepH-E3 in MANDALA | 243 | 153 | 124 | 6 | 8 |
| Graphene MACE-H native | 242 | (149, in decode) | — | — | 161 |
| Graphene MACE-H in MANDALA | 269 | 153 | 123 | 6 | 8 |
| MoS₂ DeepH-E3 native | 246 | (425, in decode) | — | — | 488 |
| MoS₂ DeepH-E3 in MANDALA | 257 | 453 | 159 | 8 | 54 |
| MoS₂ MACE-H native | 265 | (414, in decode) | — | — | 475 |
| MoS₂ MACE-H in MANDALA | 293 | 449 | 156 | 8 | 54 |
| MoS₂ DeepH native | 7.6 | (445, in decode) | — | — | 1015 |
| MoS₂ DeepH in MANDALA | 7.5 | 422 | 156 | 10 | 579 |

"Network" is `pipeline.forward` for MANDALA and the upstream `model.forward`
for native. MANDALA's network time also includes the irreps-to-packed
decoding (`native_decode`, 7 ms graphene / 13 ms MoS₂), which native does in
decode via `codec.get_H`. Allowing for that:

* **DeepH**: identical.
* **DeepH-E3**: within ±1% (236 vs 234; 244 vs 246).
* **MACE-H**: +6–8% (262 vs 242; 281 vs 265). Edge updates and angular expansions
  match native to within 1–3 ms. The excess is in the neighbour-density/product
  ("mace") stages: 64 vs 54 ms (graphene), 80 vs 71 ms (MoS₂). This is the only
  network-level difference found. It was not investigated further.

### Profiler counts for one forward (graphene)

| | GPU kernel time | GPU busy | Kernel launches | D2H syncs |
|---|---:|---:|---:|---:|
| DeepH-E3 native | 39.7 ms | 17% | 9,884 | 19 |
| DeepH-E3 in MANDALA | 94.6 ms | 18% | 20,170 | 23,506 |
| MACE-H native | 75.1 ms | 32% | 10,011 | 51 |
| MACE-H in MANDALA | 130.3 ms | 25% | 20,300 | 23,534 |

The extra ~10k kernel launches are copy kernels. Each per-edge column of
`edges.T` is non-contiguous and must be made contiguous before the host copy.
The extra GPU time is mostly those copies and the ~16.7k small device-to-host
transfers.

### Counterfactuals: vectorised replacements (ms)

These blocks run after the profiler trace. Every variant, including ones that
never call a replaced function, measured ~10–15% slower there than in the
baseline block (e.g. native DeepH-E3 forward 234 → 261 ms). That is
consistent with CUPTI tracing staying attached after `torch.profiler`.
**Compare counterfactual columns with each other, not with the baseline.**

| Variant | Graphene e2e, both fixed | MoS₂ e2e, both fixed | in-MANDALA vs native |
|---|---:|---:|---|
| DeepH native | 925 | 1738 | |
| DeepH in MANDALA | 931 | 1744 | +1% / +0% |
| DeepH-E3 native | 298 | 410 | |
| DeepH-E3 in MANDALA | 308 | 428 | +3% / +4% |
| MACE-H native | 293 | 415 | |
| MACE-H in MANDALA | 312 | 438 | +6% / +5% |

Within the counterfactual blocks, fixing only `from_packed` leaves the
in-MANDALA forward 142–184 ms above the both-fixed value: that is the
trace-alignment cost. Fixing only trace alignment leaves it 166–424 ms above:
that is the `from_packed` cost. This matches the stage timings. With both fixed,
the in-MANDALA forward is 288 / 291 ms (graphene DeepH-E3 / MACE-H) and
352 / 363 ms (MoS₂). Those figures still include output wrapping, which native
excludes from forward. The same vectorised `from_packed` also speeds up the
**native** end-to-end path. Even with the post-profiler inflation, graphene
DeepH-E3 drops from 410 to 298 ms and MoS₂ from 747 to 410 ms.

## Costs shared by both implementations (not part of the gap)

* **The network is CPU/launch bound.** GPU busy is 17–32% for the native
  E3 models with eager e3nn: ~10k launches and ~2,000 small `mm` calls at
  ~58 µs CPU each. The scripted e3nn backend stalled on first call on these
  GPUs (attempt-4 report), so it was not tested here. Earlier script-mode
  timings were only 10–25% faster for in-MANDALA variants.
* **DeepH bond-frame decoding.** `IrrepsBlockData.to_blocks` with frames does
  per-edge `.tolist()` to look up frames: 178 ms (graphene), 570 ms (MoS₂), in
  both versions. DeepH preparation (`evaluate_graph_geometry` with local
  coordinates) takes another ~400 ms ×2 on MoS₂. These dominate DeepH's 1–2 s
  end-to-end latency, against a 6–8 ms network.
* **MANDALA's own model** has none of the forward overhead: 15 syncs, because its
  dataset path builds the metadata during `prep`. Its end-to-end time is
  dominated by that preparation: 331 ms (graphene), 401 ms (MoS₂). Trace
  alignment is 28 ms and 37 ms of that (three calls per structure).

## Implications

* The benchmark's "forward" column compares different scopes. Native forward
  excludes output wrapping, while MANDALA forward includes it. Only the
  end-to-end column is a like-for-like comparison.
* The genuine end-to-end overhead (1.1–1.3×) is loss-only metadata computed
  at inference, plus per-edge synchronisations in shared output code. Both are
  fixable without changing any results. After fixing, in-MANDALA DeepH and
  DeepH-E3 are within 0–4% of native end to end, and MACE-H within 6%.
* Training is likely affected too: the same metadata construction
  runs per forward unless already cached in the batch. That was not measured here.

## Reproduce

```bash
# on ROSI, from the run directory
sbatch --partition=gpu-h100 --time=01:30:00 run.sbatch graphene
sbatch --partition=gpu-h100 --time=01:30:00 run.sbatch mos2
python summarize.py results/<graphene job> results/<mos2 job>
```

The launcher puts the benchmark's frozen framework on `PYTHONPATH` read-only.
Per-variant `result.json` files and profiler tables are in
`docs/data/inference_profile_20261111/`.
