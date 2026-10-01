# Research operations for Mandala — pair Hamiltonian

Read this file and AGENTS.md before taking tasks. Dashboard project ID: mandala-pair-hamiltonian.
Source branch: pair-descriptor-to-hamiltonian; onboarding baseline: 7531ca927d502c53dbda39233cf6b210528cf173.
Laptops use /home/bartek/casus/mandala. ROSI and terok use ~/casus/mandala. The pair-Hamiltonian branch additionally has its own terok worktree ~/casus/mandala-pair_hamiltonian. Never switch another active checkout.
Large-data root: /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian. Preserve existing datasets and split provenance; do not create a guessed alternate data root.

# ROSI project setup inspected 2026-10-01

ROSI uses ~/scripts/hpc/hpc.py, presets.toml and aliases.zsh. Preset project names are mandala, molecula, wyckoff. Profiles under ~/scripts/profiles select the module and virtualenv automatically. Account: casus. The clean profile deliberately does not activate an environment. ~/scripts/python.profile is a compatibility alias for clean, not a project environment.

Shell home /home/brzoza73 and physical home /data/home2/brzoza73 refer to the same home. Code checkouts are ~/casus/{mandala,MolecuLA,wyckoff_materials}. Mandala's current ROSI checkout is pair-descriptor-to-hamiltonian; do not switch it for a GNN job. Isolated researchops mirrors/workspaces carry the correct full commit separately.

| Project | Profile | Environment | Large artifacts |
|---|---|---|---|
| Mandala GNN | ~/scripts/profiles/mandala.sh | Python module 3.10.4; ~/casus/mandala/mandala-venv-b200-test (CUDA 12.8) | /bigdata/casus/wdm/hamiltonian_learning/data; /bigdata/casus/wdm/hamiltonian_learning/models/checkpoints |
| Mandala pair Hamiltonian | same profile, separate Git branch | same; legacy profile mandala_legacy uses mandala-venv for V100 | /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian; use a distinct output namespace |
| MolecuLA | ~/scripts/profiles/molecula.sh | Python 3.12.4 + CUDA 12.8; ~/casus/MolecuLA/molecula-venv | /bigdata/casus/wdm/BB/MolecuLA/{data,checkpoints,chemspace_runs,staging}; existing additional environments under venvs |
| Wyckoff | ~/scripts/profiles/wyckoff.sh | Python 3.12.4 + CUDA 12.8; ~/casus/wyckoff_materials/crystal-venv | /bigdata/casus/wdm/BB/wyckoff_materials/{data,cache,checkpoints,reports,runs} |

The main Wyckoff P1 dataset is /bigdata/casus/wdm/BB/wyckoff_materials/data/wyckoff_dvae_categorical_v2/mp20_clean_canonical_seed0_v2. Its geometry cache is the adjacent mp20_clean_canonical_seed0_v2_periodic_geometry directory. Existing split provenance must be preserved. Shared OpenMX references are /bigdata/casus/wdm/BB/reference/OpenMX/{VPS,PAO}, not a new Mandala data root.

Normal CLI examples from repository root:
- ~/scripts/hpc/hpc.py run --project wyckoff --gpu h100 --gpus 1 --time 08:00:00 python -m tools.<entrypoint> <explicit arguments>
- runh1/runh2/runh4/runh8 select H100; runhc* select H100 CASUS; runbc* select B200 CASUS.
- H100 preset CPU requests are 12/32/64/128 for 1/2/4/8 GPUs. B200 CASUS requests 32/64/64/256. Defaults reserve substantial memory and 48 hours; override resource limits explicitly.

For automated jobs the approved gateway uses an isolated Apptainer environment, full commit and declared runtime identity instead of relying on an interactive shell profile. Its registry pins exact public package/virtualenv read mounts and runtime library paths. No host home, personal credentials or network is given to scientific jobs. Heavy job directories go to the project's bigdata runs namespace; only the private ownership ledger and small mirrors remain in home. All settings/artifact paths belong in explicit YAML/CLI, not ambient experiment environment variables. W&B queries and interpretation run on the laptop, never the ROSI login node. New project pages start empty; old studies are not silently imported.
