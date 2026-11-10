"""Locate differences in geometric features and pipeline activations under tiling."""
import argparse

from workflow import *
from data.structure_inference import build_model_input_from_structure
from net.evaluation import _move_to_device
from fold_reference import folded_keys


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--material',required=True,choices=['silicon','sio2'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    model,cfg,_ = load_model(args.material,'cuda')
    structure = next(s for s in manifest()['structures'] if s['material']==args.material and s['tiled_control'])
    base,_ = reference(structure['source_cell'],cfg)
    atoms,pos,box = load_paper_cif(RUN/'structures'/structure['cif'],dtype=cfg.dtype)
    xb = build_model_input_from_structure(atoms=base.hamiltonian.atoms,positions=base.positions,
                                          box=base.box,cfg=cfg,mapper=model.mapper)
    xs = build_model_input_from_structure(atoms=atoms,positions=pos,box=box,cfg=cfg,mapper=model.mapper)
    old_keys = torch.cat((xb['edge_shift'],xb['edge_index'])).T.tolist()
    new_keys = torch.cat((xs['edge_shift'],xs['edge_index'])).T.tolist()
    new_lookup = {tuple(key):i for i,key in enumerate(new_keys)}
    relation = list(folded_keys(old_keys,len(base.hamiltonian.atoms),structure['repetition']))
    common = [(new_lookup[key],old) for key,old in relation if key in new_lookup]
    ni,bi = map(torch.tensor,zip(*common))
    report = dict(material=args.material,base_edges=len(old_keys),super_edges=len(new_keys),
                  expected_super_edges=len(relation),common_edges=len(common),inputs={},activations={})
    for name in ('edge_length','edge_sh','edge_length_emb','edge_one_hot','edge_type_idx'):
        delta = (xs[name][ni].double()-xb[name][bi].double()).abs()
        report['inputs'][name] = dict(mean=float(delta.mean()),max=float(delta.max()))
    values = {}
    hooks = []
    for name,module in model.pipeline.transforms.items():
        def hook(module,inputs,output,name=name):
            if torch.is_tensor(output):
                values[name] = output.detach().cpu()
        hooks.append(module.register_forward_hook(hook))
    xgpu = _move_to_device(xb,torch.device('cuda'))
    model.predict_matrices(xgpu,physical=True)
    previous = dict(values)
    values.clear()
    xgpu = _move_to_device(xs,torch.device('cuda'))
    model.predict_matrices(xgpu,physical=True)
    for name,value in values.items():
        original = previous[name]
        if len(original)==len(old_keys) and len(value)==len(new_keys):
            original,value = original[bi],value[ni]
        elif len(original)==len(base.hamiltonian.atoms) and len(value)==len(atoms):
            original = original.repeat((len(atoms)//len(original),)+(1,)*(original.ndim-1))
        else:
            report['activations'][name] = dict(base_shape=list(original.shape),super_shape=list(value.shape))
            continue
        delta = (value.double()-original.double()).abs()
        report['activations'][name] = dict(mean=float(delta.mean()),max=float(delta.max()),
                                           base_absolute_mean=float(original.abs().mean()))
    for hook in hooks:
        hook.remove()
    write_json(RUN/'results'/(args.material+'_tiling_probe.json'),report)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
