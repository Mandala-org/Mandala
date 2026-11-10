"""Audit the 600-atom extension without requiring model or operator files."""
import json
from pathlib import Path
import subprocess

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]


def main():
    baseline = json.loads(subprocess.check_output(
        ['git', 'show', 'c0d37ed:researchops/runs/r6_supercell_20261110/results/comparison.json'],
        cwd=ROOT, text=True))
    current = json.loads((RUN/'results/comparison.json').read_text())
    rows = {s['name']: s for s in current['structures']}
    checked = 0
    for old in baseline['structures']:
        if old.get('element_errors'):
            assert rows[old['name']]['element_errors'] == old['element_errors'], old['name']
            checked += 1
    control = rows['sio2-2x2x1-tiled']
    error = control['element_errors']['hamiltonian']['hermitian_common_support']['mae_meV']
    provenance = json.loads((RUN/'results/sio2_4x_provenance.json').read_text())
    source = provenance['historical_control']['current_vs_reference']['hamiltonian']['hermitian_common_support']['mae_meV']
    assert abs(error-source) < 1e-4, (error, source)
    assert control['atoms'] == 600 and control['status'] == 'ok'
    perturbed = rows['sio2-2x2x1-1']
    assert perturbed['atoms'] == 600
    if perturbed['status'] == 'OpenMX_pending':
        assert perturbed['job_id'] == '837830' and 'element_errors' not in perturbed
    else:
        assert perturbed['status'] == 'ok' and perturbed['reference_kind'] == 'new self-consistent OpenMX'
    diagnostic = json.loads((RUN/'results/sio2-2x2x1-1.json').read_text())
    assert set(diagnostic['operator_sha256']) == {'hamiltonian'}
    residual = diagnostic['predicted']['records'][0]['raw_hermiticity']['hamiltonian']['relative_frobenius']
    assert abs(residual-0.004155365412447879) < 1e-14
    summary = json.loads((RUN/'results/summary.json').read_text())
    table = {s['name']: s for s in summary['structures']}
    assert table['sio2-2x2x1-tiled']['H_MAE_meV'] == error
    if perturbed['status'] == 'OpenMX_pending':
        assert table['sio2-2x2x1-1']['H_MAE_meV'] is None
        assert table['sio2-2x2x1-1']['job_id'] == '837830'
    assert 'final_compare_sio2_4x.sh' in (RUN/'results/summary.md').read_text()
    print(f'PASS: {checked} previous element-error rows unchanged; tiled {error:.12g} meV vs source {source:.12g}; raw residual {residual:.12g}; perturbed status {perturbed["status"]}.')


if __name__ == '__main__':
    main()
