"""Quantify changed OpenMX runtime versus the same source-cell calculations."""
from workflow import *


def main():
    torch.set_num_threads(4)
    records = json.loads((RUN/'openmx/calibration.json').read_text())
    results = {}
    for material,record in records.items():
        if not (Path(record['new'])/'COMPLETE').exists():
            results[material] = dict(status='pending')
            continue
        _,original = reference(record['original'])
        _,new = reference(record['new'])
        differences = errors(new,original)
        h_mae = differences['hamiltonian']['raw_common_support']['mae_meV']
        result = dict(status='ok',element_errors=differences,
                      H_MAE_meV=h_mae,passed_H_tolerance=h_mae < .01,
                      H_tolerance_meV=.01,
                      maximum_absolute_error={})
        for name,a in new.items():
            result['maximum_absolute_error'][name] = max(float((a[key]-original[name][key]).abs().max())
                                                         for key in a.lookup if key in original[name].lookup)
        results[material] = result
    write_json(RUN/'results/runtime_calibration.json',results)
    print(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
