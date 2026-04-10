# Minimal Silicon E3MLP Sweep

This study forks the minimal silicon multi-snapshot trainer and swaps the fixed Gate-MLP head for the E3MLP variant zoo from [studies/e3mlp_investigation/variants.py](/home/bartek/casus/mandala/studies/e3mlp_investigation/variants.py).

Goal:
- train an E3GNN on `density` only
- evaluate `energy_mae` using ground-truth Hamiltonian and predicted density
- do not train on energy in this sweep
- sweep the most promising E3MLP choices for the message-passing internals and the head

Main entry points:
- trainer: [train_density_energy_minimal.py](/home/bartek/casus/mandala/studies/minimal_silicon_e3mlp_sweep/train_density_energy_minimal.py)
- network: [common.py](/home/bartek/casus/mandala/studies/minimal_silicon_e3mlp_sweep/common.py)
- sweep config: [sweep_density_energy_bayes.yaml](/home/bartek/casus/mandala/studies/minimal_silicon_e3mlp_sweep/sweep_density_energy_bayes.yaml)
- launch helper: [launch_sweep.sh](/home/bartek/casus/mandala/studies/minimal_silicon_e3mlp_sweep/launch_sweep.sh)
- agent helper: [run_agent.sh](/home/bartek/casus/mandala/studies/minimal_silicon_e3mlp_sweep/run_agent.sh)

Important behavior:
- `--matrix-targets density` is supported
- `--enable-energy true` computes `Tr(H_gt D_pred)` when Hamiltonian is not predicted
- `--train-on-energy false` in the prepared sweep
- internal E3MLPs refine node and edge hidden features after each message block
- head E3MLPs are separate for diagonal and off-diagonal edge blocks
- diagonal/off-diagonal head output scales are configurable to better match observed target scale mismatch at random init

Prepared sweep defaults are HPC-oriented:
- data path: `/bigdata/casus/wdm/hamiltonian_learning/data/silicon_very_big/dataset_A`
- snapshot cache: `/bigdata/casus/wdm/hamiltonian_learning/data/silicon_very_big/snapshot_cache`
- checkpoints: `studies/minimal_silicon_e3mlp_sweep/checkpoints`

Suggested sweep objective:
- minimize `val/energy_mae`

Suggested launch:
```bash
bash studies/minimal_silicon_e3mlp_sweep/launch_sweep.sh
```

Then start one or more agents:
```bash
bash studies/minimal_silicon_e3mlp_sweep/run_agent.sh <entity/project/sweep_id>
```
