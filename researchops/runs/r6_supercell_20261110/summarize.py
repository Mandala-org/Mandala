"""Render quoteable primary, parent-category and explicitly diagnostic results."""
import json
from pathlib import Path

RUN = Path(__file__).resolve().parent


def metric(diagnostic):
    records = diagnostic.get('records', [])
    overlap = [r['overlap'] for r in records if 'overlap' in r]
    if not overlap:
        return None
    return dict(kpoints=len(overlap), positive_definite_count=sum(r['positive_definite'] for r in overlap),
                positive_definite_fraction=sum(r['positive_definite'] for r in overlap)/len(overlap),
                usable_count=sum(r['well_conditioned_positive_definite'] for r in overlap),
                minimum_eigenvalue=min(r['lambda_min'] for r in overlap),
                cholesky_success_count=sum(not r['cholesky_failed_without_jitter'] for r in overlap),
                per_kpoint=[dict(fractional_k=r['fractional_k'], **r['overlap']) for r in records])


def fmt(value):
    return '—' if value is None else f'{value:.7g}'


def validity(value):
    if value is None:
        return 'H only; S not predicted'
    return (f"PD {value['positive_definite_count']}/{value['kpoints']}; "
            f"min {value['minimum_eigenvalue']:.7g}; chol {value['cholesky_success_count']}/{value['kpoints']}")


def main():
    manifest = json.loads((RUN/'structures/manifest.json').read_text())
    comparison = json.loads((RUN/'results/comparison.json').read_text())
    route_path = RUN/'results/parent_routing.json'
    routes = {r['name']: r for r in json.loads(route_path.read_text())['structures']} if route_path.exists() else {}
    primary = {r['name']: r for r in comparison['structures']}
    rows = []
    lines = ['| Structure | Atoms | Repetition | σ Å | H meV | H excl. same-parent image meV | Routed diagnostic H meV | S MAE | D MAE | Primary metric validity | Routed metric validity | Paper H meV |',
             '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |']
    for s in manifest['structures']:
        p = primary.get(s['name'], {})
        routed = routes.get(s['name'], {})
        def error(group, name, key):
            return group.get('element_errors', {}).get(name, {}).get('hermitian_common_support', {}).get(key)
        row = dict(name=s['name'], atoms=s['atom_count'], repetition=s['repetition'], sigma_angstrom=s['sigma_angstrom'],
                   status=p.get('status', 'pending'), job_id=p.get('job_id'), H_MAE_meV=error(p,'hamiltonian','mae_meV'),
                   H_excluding_same_parent_image_meV=p.get('category_errors',{}).get('hamiltonian',{}).get('hermitian_common_support',{}).get('excluding_same_parent_image',{}).get('mae_meV'),
                   routed_diagnostic_H_MAE_meV=error(routed,'hamiltonian','mae_meV'),
                   S_MAE=error(p,'overlap','mae'), D_MAE=error(p,'density','mae'),
                   routed_diagnostic_S_MAE=error(routed,'overlap','mae'), routed_diagnostic_D_MAE=error(routed,'density','mae'),
                   primary_metric=metric(p.get('predicted',{})), routed_diagnostic_metric=metric(routed.get('predicted',{})),
                   validation_H_MAE_meV=comparison['validation_H_MAE_meV'][s['material']],
                   category_errors=p.get('category_errors'), routed_category_errors=routed.get('categories'))
        rows.append(row)
        values = [s['name'], str(s['atom_count']), '×'.join(map(str,s['repetition'])), fmt(s['sigma_angstrom'])]
        values += [fmt(row[k]) for k in ('H_MAE_meV','H_excluding_same_parent_image_meV','routed_diagnostic_H_MAE_meV','S_MAE','D_MAE')]
        values += [validity(row['primary_metric']) if row['status']=='ok' else (row['status'] + (f" (job {row['job_id']})" if row['job_id'] else '') + ('; '+validity(row['primary_metric']) if row['primary_metric'] else '')),
                   validity(row['routed_diagnostic_metric']) if routed else '—',fmt(row['validation_H_MAE_meV'])]
        lines.append('| '+' | '.join(values)+' |')
    silicon = [r for r in rows if r['name'].startswith('si-') and not r['name'].endswith('tiled') and r['H_MAE_meV'] is not None]
    routed_si = [r['routed_diagnostic_H_MAE_meV'] for r in silicon if r['routed_diagnostic_H_MAE_meV'] is not None]
    excluded = [r['H_excluding_same_parent_image_meV'] for r in silicon if r['H_excluding_same_parent_image_meV'] is not None]
    sio2 = [r for r in rows if r['name'].startswith('sio2-') and not r['name'].endswith('tiled') and r['H_MAE_meV'] is not None]
    interpretation = (f"Silicon perturbed-cell H MAE is {min(r['H_MAE_meV'] for r in silicon):.6g}–{max(r['H_MAE_meV'] for r in silicon):.6g} meV versus 0.435164 meV on paper validation, with similar errors at 64 and 216 atoms.")
    if excluded:
        interpretation += f" Excluding same-parent-image blocks gives {min(excluded):.6g}–{max(excluded):.6g} meV."
    if routed_si:
        interpretation += f" Routing with known parent labels gives {min(routed_si):.6g}–{max(routed_si):.6g} meV; this diagnostic does not establish a deployable correction."
    if sio2:
        interpretation += ' SiO2 perturbed-cell H MAE is '+', '.join(f"{r['H_MAE_meV']:.6g}" for r in sio2)+' meV versus 1.99588 meV on validation.'
    else:
        interpretation += ' SiO2 perturbed references remain pending; its tiled-control H MAE is 2.534627 meV, so perturbed size transfer cannot yet be assessed.'
    for key, label in (('primary_metric','Primary'),('routed_diagnostic_metric','Parent-routed diagnostic')):
        metrics = [r[key] for r in silicon if r[key] is not None]
        if metrics:
            interpretation += (f" {label} overlap is positive definite at {sum(m['positive_definite_count'] for m in metrics)}/{sum(m['kpoints'] for m in metrics)} sampled perturbed-cell k-points;"
                               f" its lowest eigenvalue across those cells is {min(m['minimum_eigenvalue'] for m in metrics):.6g}.")
    interpretation += ' The metric uses full geometry support without repair, whereas element errors use common reference support.'
    meaning = ('Each row is one structure; repetition expands the source cell, and σ is the independent Gaussian displacement standard deviation (zero for tiled controls). H, S and D are scalar-entry-weighted MAEs on matching directed periodic support after Hermitian projection of the prediction, against OpenMX (D multiplied by two); H is in meV, S dimensionless, D in AO-kernel units. H excl. removes every non-onsite pair sharing a source-atom identity, including nonzero lattice shifts; onsite plus different-parent blocks remain. Routed columns use unchanged silicon heads with construction parent labels and never replace primary predictions. Metric columns give positive-definite k-point counts, global minimum overlap eigenvalue, and Cholesky successes without jitter on full geometry support, using 4×4×4 (2× Si) 3×3×3 (3× Si), or 2×2×2 (5× Si) Gamma-centered meshes. SiO2 predicts H only; — means unavailable or not predicted. Paper H is the material validation mean, not a supercell target. Category fractions below use common support, separately counting directed blocks and scalar entries; JSON also retains prediction/reference full-support fractions and raw errors. Tiled references fold source operators; the SiO2 control requires converged source Gamma sampling and the 3× and 5× Si controls fold the original 8-mesh calculation.')
    lines += ['',interpretation,'',meaning,'', '| Structure | Operator | Category | Blocks fraction | Scalar fraction | MAE (H meV) |', '| --- | --- | --- | ---: | ---: | ---: |']
    for row in rows:
        for name, groups in (row['category_errors'] or {}).items():
            for kind, values in groups['hermitian_common_support']['categories'].items():
                lines.append('| '+' | '.join([row['name'],name,kind,fmt(values['block_fraction']),fmt(values['scalar_entry_fraction']),fmt(values.get('mae_meV',values['mae']))])+' |')
    lines += ['', 'Parent-routing diagnostic (requires known parent labels):', '', '| Structure | H meV | S MAE | D MAE | PD fraction | Minimum S eigenvalue | Cholesky successes |', '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for row in rows:
        m = row['routed_diagnostic_metric']
        if m:
            lines.append('| '+' | '.join([row['name'],fmt(row['routed_diagnostic_H_MAE_meV']),fmt(row['routed_diagnostic_S_MAE']),fmt(row['routed_diagnostic_D_MAE']),fmt(m['positive_definite_fraction']),fmt(m['minimum_eigenvalue']),f"{m['cholesky_success_count']}/{m['kpoints']}"])+' |')
    lines += ['', '1000-atom follow-up: after OpenMX completes, run:', '', '```bash', 'ssh rosi4 "cd /data/home2/brzoza73/casus/mandala_worktrees/r6 && sbatch --parsable researchops/runs/r6_supercell_20261110/final_compare_5x.sh"', '```']
    lines += ['', '600-atom SiO2 follow-up: after OpenMX completes, run:', '', '```bash', 'ssh rosi4 "cd /data/home2/brzoza73/casus/mandala_worktrees/r6 && sbatch --parsable researchops/runs/r6_supercell_20261110/final_compare_sio2_4x.sh"', '```']
    (RUN/'results/summary.md').write_text('\n'.join(lines)+'\n')
    (RUN/'results/summary.json').write_text(json.dumps(dict(diagnostic_caveat='Requires known parent cell; unchanged parameters; not applied to primary predictions', interpretation=interpretation, columns=meaning, structures=rows),indent=2,allow_nan=False)+'\n')
    print('\n'.join(lines[:len(rows)+5]))


if __name__ == '__main__':
    main()
