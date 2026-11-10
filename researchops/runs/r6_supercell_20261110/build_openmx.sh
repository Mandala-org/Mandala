#!/usr/bin/env bash
#SBATCH --job-name=env-r6-openmx
#SBATCH --partition=cpu-genoa-casus
#SBATCH --account=casus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=researchops/runs/r6_supercell_20261110/openmx/build-%j.out
set -euo pipefail
source /etc/profile.d/lmod.sh
ml purge
ml genoa
ml foss/2022a
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
build_root=/bigdata/casus/wdm/hamiltonian_learning/physical_constraints/r6_supercell_20261110/environment
mkdir -p "$build_root"
cd "$build_root"
if [[ ! -f openmx3.9.tar.gz ]]; then
    curl -fL --retry 3 https://www.openmx-square.org/openmx3.9.tar.gz -o openmx3.9.tar.gz
fi
if [[ ! -f patch3.9.9.tar.gz ]]; then
    curl -fL --retry 3 https://www.openmx-square.org/bugfixed/21Oct17/patch3.9.9.tar.gz -o patch3.9.9.tar.gz
fi
echo '27bb56bd4d1582d33ad32108fb239b546bdd1bdffd6f5b739b4423da1ab93ae2  openmx3.9.tar.gz' | sha256sum -c -
echo '20cccc4e3412a814a53568f400260e90f79f0bfb7e2bed84447fe071b26edd38  patch3.9.9.tar.gz' | sha256sum -c -
sha256sum openmx3.9.tar.gz patch3.9.9.tar.gz > archives.sha256
tar xzf openmx3.9.tar.gz
cd openmx3.9/source
tar xzf ../../patch3.9.9.tar.gz
cp makefile makefile.upstream
cc_flags="mpicc -O3 -march=znver3 -fopenmp -fcommon -I${EBROOTFFTW}/include"
fc_flags="mpif90 -O3 -march=znver3 -fopenmp -fallow-argument-mismatch"
link_flags="-L${EBROOTSCALAPACK}/lib -lscalapack -L${EBROOTFLEXIBLAS}/lib -lflexiblas -L${EBROOTFFTW}/lib -lfftw3 -lmpi_usempif08 -lmpi_usempi_ignore_tkr -lmpi_mpifh -lgfortran -lm -ldl"
# The upstream makefile omits Fortran module dependencies. A parallel make
# races .mod producers/consumers; serialize it rather than changing numerics.
make -j 1 openmx analysis_example "CC=$cc_flags" "FC=$fc_flags" "LIB=$link_flags" DESTDIR=../work > build.log 2>&1 || { tail -80 build.log; exit 1; }
test -x openmx
test -x analysis_example
sha256sum openmx analysis_example ../DFT_DATA19/PAO/{Si7.0,Si9.0,O7.0}.pao ../DFT_DATA19/VPS/{Si_PBE19,O_PBE19}.vps > ../r6-runtime.sha256
ldd openmx > ../r6-ldd.txt
module list 2> ../r6-modules.txt
printf '%s\n' "$cc_flags" "$fc_flags" "$link_flags" > ../r6-build-flags.txt
date -u > ../BUILD_COMPLETE
