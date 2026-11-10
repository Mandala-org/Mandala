#!/usr/bin/env bash
#SBATCH --job-name=r6-dft-si-2x2x2-2
#SBATCH --partition=cpu-genoa-casus
#SBATCH --account=casus
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/data/home2/brzoza73/casus/mandala_worktrees/r6/researchops/runs/r6_supercell_20261110/openmx/si-2x2x2-2/slurm-%j.out
set -euo pipefail
source /etc/profile.d/lmod.sh
ml purge
ml genoa
ml foss/2022a
export OMPI_MCA_pml=ucx
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /bigdata/casus/wdm/hamiltonian_learning/data/r6_supercells_20261110/silicon/si-2x2x2-2
test ! -e COMPLETE
test -x /bigdata/casus/wdm/hamiltonian_learning/physical_constraints/r6_supercell_20261110/environment/openmx3.9/source/openmx
test -x /bigdata/casus/wdm/hamiltonian_learning/physical_constraints/r6_supercell_20261110/environment/openmx3.9/source/analysis_example
date -u
mpirun -bind-to core --map-by core -np "$SLURM_NTASKS" /bigdata/casus/wdm/hamiltonian_learning/physical_constraints/r6_supercell_20261110/environment/openmx3.9/source/openmx conf.dat > log.out
grep -q 'The calculation was normally finished.' log.out
grep -q 'The SCF was achieved' log.out
test -s Si.scfout
mpirun -bind-to core --map-by core -np 1 /bigdata/casus/wdm/hamiltonian_learning/physical_constraints/r6_supercell_20261110/environment/openmx3.9/source/analysis_example Si.scfout > HS.out.tmp
test -s HS.out.tmp
mv HS.out.tmp HS.out
# Keep scfout on any failed run. Remove bulky intermediates only after extraction.
shopt -s nullglob
intermediates=( *.md* *.cube *.scfout *.input *.dat# *rst )
if (( ${#intermediates[@]} )); then rm -rf -- "${intermediates[@]}"; fi
date -u > COMPLETE
