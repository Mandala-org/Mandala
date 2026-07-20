#!/usr/bin/env bash
set -euo pipefail

paper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="$(mktemp -d "${paper_dir}/.build.XXXXXX")"
trap 'rm -rf "${build_dir}"' EXIT

cd "${paper_dir}"
run_pdflatex() {
  local pass="$1"
  if ! pdflatex -interaction=nonstopmode -halt-on-error \
    -output-directory="${build_dir}" main.tex \
    >"${build_dir}/pdflatex-${pass}.log" 2>&1; then
    tail -n 80 "${build_dir}/pdflatex-${pass}.log"
    return 1
  fi
}

run_pdflatex 1
if ! (
  cd "${build_dir}"
  BIBINPUTS="${paper_dir}:" BSTINPUTS="${paper_dir}:" bibtex main
) >"${build_dir}/bibtex.log" 2>&1; then
  tail -n 80 "${build_dir}/bibtex.log"
  exit 1
fi
run_pdflatex 2
run_pdflatex 3

# Replace the public PDF atomically so an open viewer never observes a
# partially rewritten file while TeX is compiling it.
mv "${build_dir}/main.pdf" "${paper_dir}/main.pdf"
printf 'Built %s\n' "${paper_dir}/main.pdf"
