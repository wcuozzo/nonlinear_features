"""Fit width-4 decoders to successful, frozen XOR circuits from the focused search."""
from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F

import minimal_learned_example as base
from focused_stage2_experiment import OUT, extract_model
from width4_codec_experiment import evaluate


def unpack_decoder(parameters):
    offset = 0
    layers = []
    for n_in, n_out in [(2,4),(4,4),(4,4)]:
        count = n_in * n_out
        w = parameters[:, offset:offset+count].reshape(-1,n_out,n_in)
        b = parameters[:, offset+count:offset+count+n_out].reshape(-1,1,n_out)
        layers.append((w,b))
        offset += count+n_out
    return layers


def forward_decoder(parameters, z):
    for i,(w,b) in enumerate(unpack_decoder(parameters)):
        z = z @ w.transpose(1,2) + b
        if i < 2:
            z = z.relu()
    return z


def pack_decoder(model):
    return torch.cat([value.detach().flatten() for layer in model.net
                      if isinstance(layer,torch.nn.Linear) for value in [layer.weight,layer.bias]])


def extract_decoder(row, mean, scale):
    decoder=base.MLP(2,4,4)
    with torch.no_grad():
        for layer,(w,b) in zip([m for m in decoder.net if isinstance(m,torch.nn.Linear)],unpack_decoder(row[None])):
            layer.weight.copy_(w[0]); layer.bias.copy_(b[0,0])
        first=decoder.net[0]
        first.weight.div_(scale[None,:])
        first.bias.sub_(first.weight @ mean)
    return decoder


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=20000)
    parser.add_argument('--restarts',type=int,default=32)
    args=parser.parse_args()
    assert args.restarts % 8 == 0
    torch.set_num_threads(1)
    torch.manual_seed(314159)
    screen=json.loads((OUT/'task-only-input-screen.json').read_text())
    indices=[r['model_index'] for r in screen if r['every_pair_numerical']]
    assert indices
    task=torch.load(OUT/'task-only.pt',weights_only=True)
    targets=base.truth_table()[None].expand(len(indices)*args.restarts,-1,-1)
    candidate_indices=torch.tensor(indices).repeat_interleave(args.restarts)
    z4=task['z4'][candidate_indices]
    mean,scale=z4.mean(1),z4.std(1)
    scale[scale<1e-6]=1
    standardized=(z4-mean[:,None,:])/scale[:,None,:]
    count=len(candidate_indices)
    parameters=torch.empty(count,52)
    for w,b in unpack_decoder(parameters):
        bound=1/math.sqrt(w.shape[-1])
        w.uniform_(-bound,bound); b.uniform_(-bound,bound)
    # Alternate random initialization with copies of the original decoder,
    # on final codes standardized to the same mean/variance as initial codes.
    for i,index in enumerate(candidate_indices.tolist()):
        if i % 2:
            decoder=base.MLP(2,4,4)
            decoder.load_state_dict(task['decoders'][index//task['runs_per_code']])
            parameters[i]=pack_decoder(decoder)
    parameters.requires_grad_()
    rates=torch.tensor([.001,.003,.01,.03]).repeat_interleave(args.restarts//4).repeat(len(indices))
    m,v=torch.zeros_like(parameters),torch.zeros_like(parameters)
    best=parameters.detach().clone()
    best_correct=torch.full((count,),-1)
    best_loss=torch.full((count,),float('inf'))
    started=time.monotonic()
    history=[]
    for step in range(1,args.steps+1):
        logits=forward_decoder(parameters,standardized)
        loss=F.binary_cross_entropy_with_logits(logits,targets,reduction='none').mean((1,2))
        loss.sum().backward()
        with torch.no_grad():
            gradient=parameters.grad
            m.mul_(.9).add_(gradient,alpha=.1)
            v.mul_(.999).addcmul_(gradient,gradient,value=.001)
            rate=rates*(1 if step<args.steps*.75 else .1)
            parameters.add_(-rate[:,None]*(m/(1-.9**step))/((v/(1-.999**step)).sqrt()+1e-8))
            parameters.grad=None
            if step%250==0 or step==args.steps:
                logits=forward_decoder(parameters,standardized)
                correct=((logits>0)==targets.bool()).all(-1).sum(-1)
                actual_loss=F.binary_cross_entropy_with_logits(logits,targets,reduction='none').mean((1,2))
                improve=(correct>best_correct)|((correct==best_correct)&(actual_loss<best_loss))
                best[improve]=parameters[improve]
                best_correct[improve]=correct[improve]
                best_loss[improve]=actual_loss[improve]
                if step%2000==0 or step==args.steps:
                    record={'step':step,'perfect_decoders':int((best_correct==16).sum()),
                            'best_input_patterns':int(best_correct.max()),
                            'elapsed_seconds':round(time.monotonic()-started,1)}
                    print(json.dumps(record),flush=True); history.append(record)
    torch.save({'parameters':best,'best_correct':best_correct,'best_loss':best_loss,
                'candidate_indices':candidate_indices,'mean':mean,'scale':scale,'history':history},OUT/'decoder-search.pt')
    summary={'candidates':indices,'decoder_restarts_per_candidate':args.restarts,'steps':args.steps,
             'perfect_decoder_indices':torch.where(best_correct==16)[0].tolist(),
             'best_patterns_per_candidate':best_correct.reshape(len(indices),-1).max(1).values.tolist(),
             'history':history,'verified_full_models':[]}
    # Test one best decoder for each successful circuit; exact certificates,
    # rather than the numerical pre-screen, determine the final claim.
    for candidate in indices:
        options=torch.where((candidate_indices==candidate)&(best_correct==16))[0]
        if not len(options): continue
        row=int(options[best_loss[options].argmin()])
        encoder,d0=base.MLP(4,2,4),base.MLP(2,4,4)
        encoder.load_state_dict(task['encoders'][candidate//task['runs_per_code']])
        d0.load_state_dict(task['decoders'][candidate//task['runs_per_code']])
        transition,head=extract_model(task['parameters'][candidate])
        d4=extract_decoder(best[row],mean[row],scale[row])
        models={'encoder':encoder,'d0':d0,'transition':transition,'head':head,'d4':d4}
        checkpoint=OUT/f'candidate-{candidate}.pt'
        torch.save({k:v.state_dict() for k,v in models.items()},checkpoint)
        result=evaluate(encoder,d0,transition,head,d4)
        result['protocol']={'encoder_seed':[4,9][candidate//task['runs_per_code']],
                            'hidden_width':4,'task_model_index':candidate,
                            'task_only_then_frozen_decoder_fit':True,
                            'decoder_model_index':row,'decoder_steps':args.steps}
        (OUT/f'candidate-{candidate}.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'candidate':candidate,'task_exact':result['final_task_exact'],
                          'input_exact':result['final_input_exact'],'persistent_inputs':result['persistent_nonlinear_input_bits'],
                          'full_result_passes':result['full_result_passes']}),flush=True)
        if result['full_result_passes']:
            summary['verified_full_models'].append(candidate)
    (OUT/'decoder-search.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    main()
