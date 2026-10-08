"""Matched frozen-decoder fitting and joint refinement of the 13 XOR circuits from the XOR-training stage.

These are decoder restarts on those 13 warm starts, not fresh circuit trials.
No loss or checkpoint criterion uses linear readability.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import minimal_learned_example as base
from focused_stage2_decoder import forward_decoder, pack_decoder, unpack_decoder
from joint_reconstruction_experiment import forward_task, task_state, decoder_state

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--weights', type=float, nargs='+', default=[0., 1., 3., 10.])
    parser.add_argument('--circuit-rate-multiplier', type=float, default=.01)
    parser.add_argument('--seed', type=int, default=314159)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.steps > 0 and args.circuit_rate_multiplier > 0
    assert len(set(args.weights)) == len(args.weights) and all(w >= 0 for w in args.weights)
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / 'trained-weights.json').exists(), 'Preserve completed sweeps; use a new output.'
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(args.seed)
    initial_path = HERE / 'joint_refinement_initial_circuits.json'
    initial = json.loads(initial_path.read_text())
    circuits = initial['circuits']
    restarts = 32
    count = len(circuits) * restarts
    task0 = torch.tensor([r['parameters'] for r in circuits]).repeat_interleave(restarts, 0)
    z0base = torch.tensor([r['z0'] for r in circuits]).repeat_interleave(restarts, 0)
    z4base = torch.tensor([r['z4'] for r in circuits]).repeat_interleave(restarts, 0)
    actual_z4 = forward_task(task0, z0base)[1]
    # Bitwise equality holds with the PyTorch build that produced the saved run; other builds differ at float32 rounding level.
    circuit_scale = z4base.abs().amax(dim=(-2, -1), keepdim=True)
    assert ((actual_z4 - z4base).abs() <= 1e-5 * circuit_scale).all(), 'The frozen codes of the starting circuits must be reproduced.'
    mean, scale = z4base.mean(1), z4base.std(1)
    scale[scale < 1e-6] = 1
    initial_decoder = torch.empty(count, 52)
    for w, b in unpack_decoder(initial_decoder):
        bound = 1 / math.sqrt(w.shape[-1])
        w.uniform_(-bound, bound)
        b.uniform_(-bound, bound)
    # Preserve the saved run's initialization order and RNG consumption exactly.
    for i in range(count):
        if i % 2:
            decoder = base.MLP(2, 4, 4)
            decoder.load_state_dict({k: torch.tensor(v) for k, v in initial['d0'].items()})
            initial_decoder[i] = pack_decoder(decoder)
    arms = len(args.weights)
    task = task0.repeat(arms, 1).requires_grad_()
    decoder = initial_decoder.repeat(arms, 1).requires_grad_()
    z0 = z0base.repeat(arms, 1, 1)
    frozen_z4 = z4base.repeat(arms, 1, 1)
    mean, scale = mean.repeat(arms, 1), scale.repeat(arms, 1)
    lam = torch.tensor(args.weights).repeat_interleave(count)
    rates = torch.tensor([.001, .003, .01, .03]).repeat_interleave(restarts // 4).repeat(len(circuits) * arms)
    x = base.truth_table()[None].expand(len(task), -1, -1)
    y = base.xor_targets(base.truth_table())[None].expand(len(task), -1, -1)
    protocol = {
        'experiment': '32 matched decoder restarts on each of 13 historical XOR circuits',
        'encoder_seeds': [9], 'reconstruction_weights': args.weights,
        'total_training_runs': len(task), 'source_circuits': [r['source_circuit'] for r in circuits],
        'decoder_restarts_per_circuit_per_weight': restarts, 'steps': args.steps, 'seed': args.seed,
        'dtype': 'float32', 'optimizer': 'Adam', 'decoder_learning_rates': [.001, .003, .01, .03],
        'circuit_rate_multiplier': args.circuit_rate_multiplier,
        'schedule': 'Initial learning rates for first 75% of steps; one tenth thereafter',
        'positive_weight_objective': 'mean XOR BCE + lambda * mean four-bit reconstruction BCE',
        'zero_weight_control': 'Circuit frozen; decoder fits the original fixed final codes using reconstruction BCE',
        'decoder_standardization': 'Fixed means and standard deviations of each original circuit final code, shared across arms and folded into exported decoder',
        'checkpoint_selection': 'Frozen control: maximize reconstructed patterns, then minimize reconstruction loss. Joint: maximize patterns with all six outputs correct, then total correct task-patterns, then minimize joint loss. Every 250 steps including endpoints; no readability term.',
        'paired_arms': 'Identical initial circuit, decoder, learning-rate assignment and decoder normalization',
        'architecture': {'encoder': [4,4,4,2], 'decoders': [2,4,4,4], 'residual_blocks': 4, 'state_width': 2, 'branch_width': 2},
        'initial_weights_sha256': hashlib.sha256(initial_path.read_bytes()).hexdigest(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (args.output/'training-source.py').write_bytes(Path(__file__).read_bytes())
    moments = [(torch.zeros_like(p), torch.zeros_like(p)) for p in (task, decoder)]
    best_loss = torch.full((len(task),), float('inf'))
    best_score = torch.full((len(task),), -1, dtype=torch.int64)
    best_step = torch.zeros(len(task), dtype=torch.int64)
    best_task, best_decoder = task.detach().clone(), decoder.detach().clone()
    history = []
    started = time.monotonic()

    def losses():
        logits, computed_z4 = forward_task(task, z0)
        z4 = torch.where((lam == 0)[:, None, None], frozen_z4, computed_z4)
        reconstructed = forward_decoder(decoder, (z4-mean[:,None,:])/scale[:,None,:])
        task_loss = F.binary_cross_entropy_with_logits(logits, y, reduction='none').mean((1,2))
        rec_loss = F.binary_cross_entropy_with_logits(reconstructed, x, reduction='none').mean((1,2))
        objective = torch.where(lam == 0, rec_loss, task_loss+lam*rec_loss)
        return objective, logits, reconstructed

    for step in range(args.steps+1):
        if step:
            losses()[0].sum().backward()
            with torch.no_grad():
                lr = rates * (1 if step < args.steps*.75 else .1)
                for parameter, (m,v), multiplier in zip((task,decoder), moments,
                    ((lam > 0)*args.circuit_rate_multiplier, torch.ones_like(lam))):
                    m.mul_(.9).add_(parameter.grad, alpha=.1)
                    v.mul_(.999).addcmul_(parameter.grad, parameter.grad, value=.001)
                    parameter.add_(-(lr*multiplier)[:,None]*(m/(1-.9**step))/((v/(1-.999**step)).sqrt()+1e-8))
                    parameter.grad = None
        if step % 250 == 0 or step == args.steps:
            with torch.no_grad():
                objective, logits, rec = losses()
                assert torch.isfinite(objective).all()
                xor_ok = ((logits > 0) == y.bool()).all(-1)
                rec_ok = ((rec > 0) == x.bool()).all(-1)
                score = torch.where(lam == 0, rec_ok.sum(-1),
                                    33*(xor_ok & rec_ok).sum(-1)+xor_ok.sum(-1)+rec_ok.sum(-1))
                improve = (score > best_score) | ((score == best_score) & (objective < best_loss))
                best_score[improve] = score[improve]
                best_loss[improve] = objective[improve]
                best_task[improve] = task[improve]
                best_decoder[improve] = decoder[improve]
                best_step[improve] = step
                if step % 2000 == 0 or step == args.steps:
                    logits, z4 = forward_task(best_task, z0)
                    z4 = torch.where((lam == 0)[:,None,None], frozen_z4, z4)
                    rec = forward_decoder(best_decoder, (z4-mean[:,None,:])/scale[:,None,:])
                    xor_good = ((logits > 0) == y.bool()).all((1,2))
                    rec_good = ((rec > 0) == x.bool()).all((1,2))
                    record = {'step': step, 'elapsed_seconds': round(time.monotonic()-started,2),
                              'by_weight': [{'lambda': weight,
                                  'xor_correct': int(xor_good[k*count:(k+1)*count].sum()),
                                  'reconstruction_correct': int(rec_good[k*count:(k+1)*count].sum()),
                                  'both_correct': int((xor_good & rec_good)[k*count:(k+1)*count].sum())}
                                  for k,weight in enumerate(args.weights)]}
                    history.append(record);print(json.dumps(record),flush=True)
    # Retain unexported parameters for direct replay comparisons; portable JSON
    # below is sufficient for auditing and does not require this ignored .pt file.
    torch.save({'task':best_task,'decoder':best_decoder,'step':best_step}, args.output/'raw-selected.pt')
    exported = best_decoder.clone()
    w,b = unpack_decoder(exported)[0]
    w.div_(scale[:,None,:]); b.sub_(mean[:,None,:] @ w.transpose(1,2))
    rows=[]
    for i in range(len(task)):
        local=i%count; source=circuits[local//restarts]['source_circuit']
        transition,head=task_state(best_task[i])
        rows.append({'id':i,'matched_run':local,'encoder_seed':9,'lambda':args.weights[i//count],
            'learning_rate':float(rates[i]),'branch_scale':[.1,.3,.7,1.][source%4],
            'decoder_initialization':'D0' if local%2 else 'random','checkpoint_step':int(best_step[i]),
            'selection_loss':float(best_loss[i]),'source_circuit':source,'decoder_restart':local%restarts,
            'circuit_rate_multiplier':0. if args.weights[i//count]==0 else args.circuit_rate_multiplier,
            'transition':transition,'head':head,'d4':decoder_state(exported[i])})
    saved={'protocol':protocol,'encoders':{'9':{'encoder':initial['encoder'],'d0':initial['d0']}},'runs':rows}
    (args.output/'trained-weights.json').write_text(json.dumps(saved,indent=2)+'\n')
    (args.output/'training-history.json').write_text(json.dumps(history,indent=2)+'\n')
    print(f'Saved {len(rows)} decoder attempts across {len(circuits)} XOR-circuit warm starts.',flush=True)


if __name__=='__main__':
    main()
