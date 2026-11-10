#!/usr/bin/env bash
#SBATCH --job-name=r6-dft-sio2-2x1x1-2
#SBATCH --partition=cpu-genoa-casus
#SBATCH --account=casus
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=12
#SBATCH --mem=768G
#SBATCH --time=44:00:00
#SBATCH --output=/data/home2/brzoza73/casus/mandala_worktrees/r6/researchops/runs/r6_supercell_20261110/openmx/sio2-2x1x1-2/slurm-%j.out
set -euo pipefail
source /etc/profile.d/lmod.sh
ml purge
ml genoa
ml foss/2022a
export OMPI_MCA_pml=ucx
export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /bigdata/casus/wdm/hamiltonian_learning/data/r6_supercells_20261110/sio2/sio2-2x1x1-2
test ! -e COMPLETE
test -x /bigdata/casus/wdm/hamiltonian_learning/physical_constraints/r6_supercell_20261110/environment/openmx3.9/source/openmx
test -x /bigdata/casus/wdm/hamiltonian_learning/physical_constraints/r6_supercell_20261110/environment/openmx3.9/source/analysis_example
date -u
mpirun -bind-to core --map-by slot:PE=12 -np "$SLURM_NTASKS" /bigdata/casus/wdm/hamiltonian_learning/physical_constraints/r6_supercell_20261110/environment/openmx3.9/source/openmx SiO2.dat > log.out
grep -q 'The calculation was normally finished.' log.out
grep -q 'The SCF was achieved' log.out
test -s SiO2.scfout
mpirun -bind-to core --map-by core -np 1 /bigdata/casus/wdm/hamiltonian_learning/physical_constraints/r6_supercell_20261110/environment/openmx3.9/source/analysis_example SiO2.scfout > HS.out.tmp
test -s HS.out.tmp
mv HS.out.tmp HS.out
# Keep scfout on any failed run. Remove bulky intermediates only after extraction.
shopt -s nullglob
intermediates=( *.md* *.cube *.scfout *.input *.dat# *rst )
if (( ${#intermediates[@]} )); then rm -rf -- "${intermediates[@]}"; fi
date -u > COMPLETE
