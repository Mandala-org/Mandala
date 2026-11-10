"""Check computed categories conserve common-support counts and absolute errors."""
import json
from pathlib import Path
import math

RUN = Path(__file__).resolve().parent


def close(a, b):
    assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-13), (a,b)


def check(groups, error):
    categories = groups['categories']
    entries = sum(v['scalar_entries'] for v in categories.values())
    assert entries == error['common_scalar_entries'] == groups['common_scalar_entries']
    assert sum(v['blocks'] for v in categories.values()) == groups['common_blocks']
    close(sum(v['absolute_error_sum'] for v in categories.values())/entries,error['mae'])
    close(sum(v['block_fraction'] for v in categories.values()),1.)
    close(sum(v['scalar_entry_fraction'] for v in categories.values()),1.)
    keep = [categories[k] for k in ('onsite','different_parent')]
    count = sum(v['scalar_entries'] for v in keep)
    assert count == groups['excluding_same_parent_image']['scalar_entries']
    close(sum(v['absolute_error_sum'] for v in keep)/count,groups['excluding_same_parent_image']['mae'])


def main():
    primary = json.loads((RUN/'results/comparison.json').read_text())
    routed = json.loads((RUN/'results/parent_routing.json').read_text())
    names = json.loads((RUN/'structures/manifest.json').read_text())['structures']
    expected = {s['name'] for s in names if s['material']=='silicon'}
    assert {r['name'] for r in routed['structures']} == expected
    checks = 0
    for row in primary['structures']:
        if row['status'] != 'ok':
            continue
        for name, modes in row['category_errors'].items():
            for mode, groups in modes.items():
                check(groups, row['element_errors'][name][mode])
                checks += 1
    for row in routed['structures']:
        for name, groups in row['categories'].items():
            check(groups, row['element_errors'][name]['hermitian_common_support'])
            checks += 1
    summary = json.loads((RUN/'results/summary.json').read_text())
    assert len(summary['structures']) == len(names) == 11
    for row in summary['structures']:
        if row['name'] in expected:
            for key in ('primary_metric','routed_diagnostic_metric'):
                m = row[key]
                n = 64 if row['atoms']==64 else 27
                assert m['kpoints'] == len(m['per_kpoint']) == n
                assert m['positive_definite_count'] == sum(k['positive_definite'] for k in m['per_kpoint'])
                assert m['cholesky_success_count'] == sum(not k['cholesky_failed_without_jitter'] for k in m['per_kpoint'])
    print(f'PASS: {checks} operator/mode decompositions conserve counts and errors; 8 routed structures; 11 summary rows; all per-k metrics preserved')


if __name__ == '__main__':
    main()
