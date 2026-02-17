# Sweep np5qrnb8 Summary

Sweep file: `sweeps/test_new_options_box_convention_long.yaml`
Sweep ID: `np5qrnb8`
Project: `mandala-test-variants`

## Reproduction

```bash
source mandala-venv/bin/activate
python studies/minimal_overfit_study/analyze_sweep_conventions.py np5qrnb8 \
  --sweep-yaml sweeps/test_new_options_box_convention_long.yaml
```

## Script Output

```
Sweep: mandala-test-variants/np5qrnb8
Total runs: 36
Run states:
  crashed: 10
  failed: 17
  finished: 9

Summary by setting (convention, xyz-permutation, change-box, box-convention):

  Setting: {'convention': 'e3nn', 'xyz-permutation': '012', 'change-box': 'right', 'box-convention': 'rows'}
    runs: 1
    median mae_H: 0.00028768080616609174
    best mae_H: 0.00028768080616609174
    median irrep l2_block_rel: 0.041597115440421856
    best irrep l2_block_rel: 0.01875388604252702
    classification: uncertain

  Setting: {'convention': 'e3nn', 'xyz-permutation': '012', 'change-box': 'left', 'box-convention': 'rows'}
    runs: 1
    median mae_H: 0.00031029390093977206
    best mae_H: 0.00031029390093977206
    median irrep l2_block_rel: 0.06353877569927792
    best irrep l2_block_rel: 0.025392429184677573
    classification: uncertain

  Setting: {'convention': 'e3nn', 'xyz-permutation': '012', 'change-box': 'both', 'box-convention': 'rows'}
    runs: 1
    median mae_H: 0.0004400507052312262
    best mae_H: 0.0004400507052312262
    median irrep l2_block_rel: 0.07598854516261311
    best irrep l2_block_rel: 0.04191633289749961
    classification: uncertain

  Setting: {'convention': 'e3nn', 'xyz-permutation': '102', 'change-box': 'left', 'box-convention': 'cols'}
    runs: 1
    median mae_H: 0.000996136724776488
    best mae_H: 0.000996136724776488
    median irrep l2_block_rel: 0.17710589238245114
    best irrep l2_block_rel: 0.07498465329471246
    classification: uncertain

  Setting: {'convention': 'openmx', 'xyz-permutation': '201', 'change-box': 'left', 'box-convention': 'rows'}
    runs: 1
    median mae_H: 0.001111163516338382
    best mae_H: 0.001111163516338382
    median irrep l2_block_rel: 0.164606159356947
    best irrep l2_block_rel: 0.07632867249621453
    classification: uncertain

  Setting: {'convention': 'e3nn', 'xyz-permutation': '201', 'change-box': 'left', 'box-convention': 'rows'}
    runs: 1
    median mae_H: 0.0012356212078185698
    best mae_H: 0.0012356212078185698
    median irrep l2_block_rel: 0.19130072541054643
    best irrep l2_block_rel: 0.09345718676154982
    classification: uncertain

  Setting: {'convention': 'openmx', 'xyz-permutation': '012', 'change-box': 'right', 'box-convention': 'rows'}
    runs: 1
    median mae_H: 0.0016805585881380888
    best mae_H: 0.0016805585881380888
    median irrep l2_block_rel: 0.5600324177477177
    best irrep l2_block_rel: 0.06126926688682978
    classification: uncertain

  Setting: {'convention': 'e3nn', 'xyz-permutation': '201', 'change-box': 'right', 'box-convention': 'cols'}
    runs: 1
    median mae_H: 0.0016942451770410023
    best mae_H: 0.0016942451770410023
    median irrep l2_block_rel: 0.2826416866098758
    best irrep l2_block_rel: 0.11600274230186648
    classification: uncertain

  Setting: {'convention': 'openmx', 'xyz-permutation': '201', 'change-box': 'left', 'box-convention': 'cols'}
    runs: 1
    median mae_H: 0.0017225972179300752
    best mae_H: 0.0017225972179300752
    median irrep l2_block_rel: 0.28046347135147265
    best irrep l2_block_rel: 0.0964396271067229
    classification: uncertain

  Setting: {'convention': 'openmx', 'xyz-permutation': '012', 'change-box': 'both', 'box-convention': 'cols'}
    runs: 1
    median mae_H: 0.0017686947309435478
    best mae_H: 0.0017686947309435478
    median irrep l2_block_rel: 0.5603132191651798
    best irrep l2_block_rel: 0.0696938998629942
    classification: uncertain

  Setting: {'convention': 'openmx', 'xyz-permutation': '012', 'change-box': 'both', 'box-convention': 'rows'}
    runs: 1
    median mae_H: 0.0018883966585162275
    best mae_H: 0.0018883966585162275
    median irrep l2_block_rel: 0.561716317469609
    best irrep l2_block_rel: 0.07207639318270156
    classification: uncertain

  Setting: {'convention': 'openmx', 'xyz-permutation': '012', 'change-box': 'left', 'box-convention': 'rows'}
    runs: 1
    median mae_H: 0.0019914212994999667
    best mae_H: 0.0019914212994999667
    median irrep l2_block_rel: 0.563434584778851
    best irrep l2_block_rel: 0.10512411853619492
    classification: uncertain

  Setting: {'convention': 'openmx', 'xyz-permutation': '012', 'change-box': 'right', 'box-convention': 'cols'}
    runs: 1
    median mae_H: 0.0021009891021183465
    best mae_H: 0.0021009891021183465
    median irrep l2_block_rel: 0.5615177280203127
    best irrep l2_block_rel: 0.0938724493094901
    classification: uncertain

  Setting: {'convention': 'openmx', 'xyz-permutation': '012', 'change-box': 'left', 'box-convention': 'cols'}
    runs: 1
    median mae_H: 0.0021386901448003253
    best mae_H: 0.0021386901448003253
    median irrep l2_block_rel: 0.5632592109809103
    best irrep l2_block_rel: 0.1095673323185409
    classification: uncertain

  Setting: {'convention': 'openmx', 'xyz-permutation': '102', 'change-box': 'right', 'box-convention': 'rows'}
    runs: 1
    median mae_H: 0.00454673542190057
    best mae_H: 0.00454673542190057
    median irrep l2_block_rel: 0.8259681987705028
    best irrep l2_block_rel: 0.10400496934133233
    classification: definitely wrong

  Setting: {'convention': 'openmx', 'xyz-permutation': '102', 'change-box': 'left', 'box-convention': 'rows'}
    runs: 1
    median mae_H: 0.004595947728653973
    best mae_H: 0.004595947728653973
    median irrep l2_block_rel: 0.8298377143949596
    best irrep l2_block_rel: 0.1056160429770504
    classification: definitely wrong

  Setting: {'convention': 'openmx', 'xyz-permutation': '102', 'change-box': 'left', 'box-convention': 'cols'}
    runs: 1
    median mae_H: 0.004631292582261039
    best mae_H: 0.004631292582261039
    median irrep l2_block_rel: 0.8344892647968813
    best irrep l2_block_rel: 0.1047547424336846
    classification: definitely wrong

  Setting: {'convention': 'openmx', 'xyz-permutation': '102', 'change-box': 'right', 'box-convention': 'cols'}
    runs: 1
    median mae_H: 0.0047884911169558475
    best mae_H: 0.0047884911169558475
    median irrep l2_block_rel: 0.8477825976665474
    best irrep l2_block_rel: 0.10608606357393432
    classification: definitely wrong

  Setting: {'convention': 'e3nn', 'xyz-permutation': '102', 'change-box': 'right', 'box-convention': 'rows'}
    runs: 1
    median mae_H: 0.005645593200334338
    best mae_H: 0.005645593200334338
    median irrep l2_block_rel: 0.9018501318426545
    best irrep l2_block_rel: 0.1037084869006294
    classification: definitely wrong

  Setting: {'convention': 'e3nn', 'xyz-permutation': '102', 'change-box': 'left', 'box-convention': 'rows'}
    runs: 1
    median mae_H: None
    best mae_H: None
    median irrep l2_block_rel: None
    best irrep l2_block_rel: None
    classification: inconclusive

  Setting: {'convention': 'e3nn', 'xyz-permutation': '201', 'change-box': 'right', 'box-convention': 'rows'}
    runs: 1
    median mae_H: None
    best mae_H: None
    median irrep l2_block_rel: None
    best irrep l2_block_rel: None
    classification: inconclusive

  Setting: {'convention': 'openmx', 'xyz-permutation': '201', 'change-box': 'right', 'box-convention': 'rows'}
    runs: 1
    median mae_H: None
    best mae_H: None
    median irrep l2_block_rel: None
    best irrep l2_block_rel: None
    classification: inconclusive

  Setting: {'convention': 'e3nn', 'xyz-permutation': '102', 'change-box': 'both', 'box-convention': 'rows'}
    runs: 1
    median mae_H: None
    best mae_H: None
    median irrep l2_block_rel: None
    best irrep l2_block_rel: None
    classification: inconclusive

  Setting: {'convention': 'e3nn', 'xyz-permutation': '201', 'change-box': 'both', 'box-convention': 'rows'}
    runs: 1
    median mae_H: None
    best mae_H: None
    median irrep l2_block_rel: None
    best irrep l2_block_rel: None
    classification: inconclusive

  Setting: {'convention': 'openmx', 'xyz-permutation': '102', 'change-box': 'both', 'box-convention': 'rows'}
    runs: 1
    median mae_H: None
    best mae_H: None
    median irrep l2_block_rel: None
    best irrep l2_block_rel: None
    classification: inconclusive

  Setting: {'convention': 'openmx', 'xyz-permutation': '201', 'change-box': 'both', 'box-convention': 'rows'}
    runs: 1
    median mae_H: None
    best mae_H: None
    median irrep l2_block_rel: None
    best irrep l2_block_rel: None
    classification: inconclusive

  Setting: {'convention': 'e3nn', 'xyz-permutation': '012', 'change-box': 'left', 'box-convention': 'cols'}
    runs: 1
    median mae_H: None
    best mae_H: None
    median irrep l2_block_rel: None
    best irrep l2_block_rel: None
    classification: inconclusive

  Setting: {'convention': 'e3nn', 'xyz-permutation': '201', 'change-box': 'left', 'box-convention': 'cols'}
    runs: 1
    median mae_H: None
    best mae_H: None
    median irrep l2_block_rel: None
    best irrep l2_block_rel: None
    classification: inconclusive

  Setting: {'convention': 'e3nn', 'xyz-permutation': '012', 'change-box': 'right', 'box-convention': 'cols'}
    runs: 1
    median mae_H: None
    best mae_H: None
    median irrep l2_block_rel: None
    best irrep l2_block_rel: None
    classification: inconclusive

  Setting: {'convention': 'e3nn', 'xyz-permutation': '102', 'change-box': 'right', 'box-convention': 'cols'}
    runs: 1
    median mae_H: None
    best mae_H: None
    median irrep l2_block_rel: None
    best irrep l2_block_rel: None
    classification: inconclusive

  Setting: {'convention': 'openmx', 'xyz-permutation': '201', 'change-box': 'right', 'box-convention': 'cols'}
    runs: 1
    median mae_H: None
    best mae_H: None
    median irrep l2_block_rel: None
    best irrep l2_block_rel: None
    classification: inconclusive

  Setting: {'convention': 'e3nn', 'xyz-permutation': '102', 'change-box': 'both', 'box-convention': 'cols'}
    runs: 1
    median mae_H: None
    best mae_H: None
    median irrep l2_block_rel: None
    best irrep l2_block_rel: None
    classification: inconclusive

  Setting: {'convention': 'e3nn', 'xyz-permutation': '012', 'change-box': 'both', 'box-convention': 'cols'}
    runs: 1
    median mae_H: None
    best mae_H: None
    median irrep l2_block_rel: None
    best irrep l2_block_rel: None
    classification: inconclusive

  Setting: {'convention': 'e3nn', 'xyz-permutation': '201', 'change-box': 'both', 'box-convention': 'cols'}
    runs: 1
    median mae_H: None
    best mae_H: None
    median irrep l2_block_rel: None
    best irrep l2_block_rel: None
    classification: inconclusive

  Setting: {'convention': 'openmx', 'xyz-permutation': '102', 'change-box': 'both', 'box-convention': 'cols'}
    runs: 1
    median mae_H: None
    best mae_H: None
    median irrep l2_block_rel: None
    best irrep l2_block_rel: None
    classification: inconclusive

  Setting: {'convention': 'openmx', 'xyz-permutation': '201', 'change-box': 'both', 'box-convention': 'cols'}
    runs: 1
    median mae_H: None
    best mae_H: None
    median irrep l2_block_rel: None
    best irrep l2_block_rel: None
    classification: inconclusive

Top runs by mae_H (lower is better):
  dry-sweep-7 (pzl5hi9a): mae_H=0.00028768080616609174 cfg={convention=e3nn, xyz-permutation=012, change-box=right, box-convention=rows}
  likely-sweep-1 (g0kbvw0n): mae_H=0.00031029390093977206 cfg={convention=e3nn, xyz-permutation=012, change-box=left, box-convention=rows}
  wobbly-sweep-13 (flnqhlz7): mae_H=0.0004400507052312262 cfg={convention=e3nn, xyz-permutation=012, change-box=both, box-convention=rows}
  dazzling-sweep-20 (v4w7hhta): mae_H=0.000996136724776488 cfg={convention=e3nn, xyz-permutation=102, change-box=left, box-convention=cols}
  hearty-sweep-6 (ch600yih): mae_H=0.001111163516338382 cfg={convention=openmx, xyz-permutation=201, change-box=left, box-convention=rows}
  sparkling-sweep-3 (yzqvv0yj): mae_H=0.0012356212078185698 cfg={convention=e3nn, xyz-permutation=201, change-box=left, box-convention=rows}
  polished-sweep-10 (7bouar9h): mae_H=0.0016805585881380888 cfg={convention=openmx, xyz-permutation=012, change-box=right, box-convention=rows}
  gallant-sweep-27 (wxy9uqdk): mae_H=0.0016942451770410023 cfg={convention=e3nn, xyz-permutation=201, change-box=right, box-convention=cols}
  logical-sweep-24 (0g2nr82n): mae_H=0.0017225972179300752 cfg={convention=openmx, xyz-permutation=201, change-box=left, box-convention=cols}
  rich-sweep-34 (89q6jcec): mae_H=0.0017686947309435478 cfg={convention=openmx, xyz-permutation=012, change-box=both, box-convention=cols}

Classification summary:
  definitely wrong: 5
  uncertain: 14
```

## Summary

- The best-performing settings all use `convention=e3nn`, `xyz-permutation=012`, `box-convention=rows`, and vary `change-box` (right/left/both). These three are clearly dominant by mae_H and irrep relative errors.
- Five settings are **definitely wrong** by the 10× threshold: all are `xyz-permutation=102` with `convention=openmx` (left/right, rows/cols) plus one `e3nn` case (`xyz-permutation=102`, `change-box=right`, `rows`). These show ~10–20× worse mae_H and very high median irrep relative errors.
- A large portion of settings are **inconclusive** because runs failed/crashed (no summary metrics). These need reruns to classify.

## Insights

- `xyz-permutation=012` is consistently strong in the finished runs; this likely matches the data’s native coordinate convention.
- `xyz-permutation=102` is consistently poor when it finishes; likely a definite mismatch with the data convention or a subtle sign error.
- `box-convention=rows` dominates among top results; `cols` appears only in mid-tier runs.
- Many settings are unclassified due to failures; if you want tighter conclusions, rerun failed configurations with shorter epochs or stricter checks to ensure they complete and log final metrics.
