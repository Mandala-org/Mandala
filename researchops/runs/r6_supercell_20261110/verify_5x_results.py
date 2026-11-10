"""Check saved 1000-atom result accounting and preserve the earlier experiment."""
import json
import math
from pathlib import Path
import subprocess

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]


def main():
    comparison = json.loads((RUN/'results/comparison.json').read_text())
    primary = {r['name']: r for r in comparison['structures']}
    routed = {r['name']: r for r in json.loads((RUN/'results/parent_routing.json').read_text())['structures']}
    previous_path = str((RUN/'results/comparison.json').relative_to(ROOT))
    previous = json.loads(subprocess.check_output(['git', 'show', 'a9f214c:'+previous_path], cwd=ROOT, text=True))
    unchanged = 0
    for row in previous['structures']:
        if row.get('status') == 'ok':
            assert primary[row['name']]['element_errors'] == row['element_errors'], row['name']
            unchanged += 1
    checked = 0
    for name in ('si-5x5x5-1', 'si-5x5x5-tiled'):
        for row, groups in ((primary[name], primary[name].get('category_errors', {})),
                            (routed[name], routed[name].get('categories', {}))):
            assert row['predicted']['mesh'] == [2, 2, 2]
            assert row['predicted']['nominal_electrons'] == 4000
            assert len(row['predicted']['records']) == 8
            for operator, group in groups.items():
                variants = group if 'raw_common_support' in group else {'hermitian_common_support': group}
                for variant, values in variants.items():
                    categories = list(values['categories'].values())
                    assert sum(v['blocks'] for v in categories) == values['common_blocks']
                    assert sum(v['scalar_entries'] for v in categories) == values['common_scalar_entries']
                    assert math.isclose(sum(v['block_fraction'] for v in categories), 1)
                    assert math.isclose(sum(v['scalar_entry_fraction'] for v in categories), 1)
                    mae = sum(v['absolute_error_sum'] for v in categories)/values['common_scalar_entries']
                    assert math.isclose(mae, row['element_errors'][operator][variant]['mae'], rel_tol=1e-12)
                    remaining = [v for k, v in values['categories'].items() if k != 'same_parent_image']
                    excluded = sum(v['absolute_error_sum'] for v in remaining)/sum(v['scalar_entries'] for v in remaining)
                    assert math.isclose(excluded, values['excluding_same_parent_image']['mae'], rel_tol=1e-12)
                    checked += 1
        if primary[name]['status'] == 'OpenMX_pending':
            assert primary[name]['job_id'] == '837489'
            assert 'element_errors' not in primary[name]
    print(f'PASS: {unchanged} previous element-error rows exactly unchanged; {checked} new operator/category decompositions; all four 8-point metric records retained')


if __name__ == '__main__':
    main()
