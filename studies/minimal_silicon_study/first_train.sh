source mandala-venv/bin/activate
python studies/minimal_silicon_study/train_silicon_minimal.py \
  --data-path /bigdata/casus/wdm/hamiltonian_learning/data/silicon_very_big/dataset_A \
  --val-temp 2700 \
  --train-temps 2700 \
  --n-snapshots-per-temp 10 \
  --val-n-snapshots 2 \
  --log-interval 10 \
  --adaptive-log-interval \
  --benchmark \
  --log-data \
  --log-model \
  --log-forward \
  --verbose-forward \
  --log-per-irrep-metrics \
  --log-per-irrep-images \
  --generate-video \
  --num-epochs 2000 \
  --run-name silicon_test
