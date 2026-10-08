"""Exact real-arithmetic evaluation of a trained circuit with dyadic saved weights.

Float32 matrix products can add tiny extra directions to an otherwise affine branch.
This audit evaluates the mathematical ReLU network exactly, rather than interpreting
floating-point roundoff as an additional nonlinear operation.
"""
import argparse
from fractions import Fraction
import hashlib
import itertools
import json

import numpy as np
import torch

from focused_stage2_experiment import OUT
from width4_codec_experiment import certify
from verify_width4_codec_experiment import verify_certificate


def linear(rows, weight, bias):
    return [[sum(a*b for a,b in zip(row,w))+c for w,c in zip(weight,bias)] for row in rows]


def relu(rows):
    return [[max(Fraction(0),v) for v in row] for row in rows]


def mlp(rows, weights):
    for i in [0,2,4]:
        rows=linear(rows,weights[f'net.{i}.weight'],weights[f'net.{i}.bias'])
        if i<4: rows=relu(rows)
    return rows


def convert(tensor):
    if tensor.ndim==1: return [Fraction(v) for v in tensor.tolist()]
    return [[Fraction(v) for v in row] for row in tensor.tolist()]


def predictions(rows):
    return [[int(v>0) for v in row] for row in rows]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',type=int,default=71)
    args=parser.parse_args()
    checkpoint=OUT/f'candidate-{args.candidate}.pt'
    saved=torch.load(checkpoint,weights_only=True)
    weights={name:{key:convert(value) for key,value in model.items()} for name,model in saved.items()}
    inputs=list(map(list,itertools.product((0,1),repeat=4)))
    targets=[[a^b,c^d] for a,b,c,d in inputs]
    z0=mlp(inputs,weights['encoder'])
    assert predictions(mlp(z0,weights['d0']))==inputs
    z=z0
    residuals=[z0]
    interfaces=[]
    for block in range(4):
        w=weights['transition']
        h=relu(linear(z,w[f'ins.{block}.weight'],w[f'ins.{block}.bias']))
        live=[a+b for a,b in zip(z,h)]
        numerical=np.array([[float(v) for v in row] for row in live])
        exact=[[str(v) for v in row] for row in live]
        certificates=[certify(numerical,np.array([row[j] for row in inputs]),exact_values=exact) for j in range(4)]
        for j,c in enumerate(certificates): verify_certificate(exact,[row[j] for row in inputs],c)
        interfaces.append({'name':f'z{block}+h{block+1}','exact_activations':exact,
                           'input_bits_linearly_separable':[c['separable'] for c in certificates],
                           'certificates':certificates})
        update=linear(h,w[f'outs.{block}.weight'],w[f'outs.{block}.bias'])
        z=[[a+b for a,b in zip(row,change)] for row,change in zip(z,update)]
        residuals.append(z)
    head=weights['head']
    xor_logits=linear(z,head['weight'],head['bias'])
    input_logits=mlp(z,weights['d4'])
    assert predictions(xor_logits)==targets
    assert predictions(input_logits)==inputs
    xor_audits=[]
    for i,state in enumerate(residuals):
        values=np.array([[float(v) for v in row] for row in state])
        exact=[[str(v) for v in row] for row in state]
        certs=[certify(values,np.array([row[j] for row in targets]),exact_values=exact) for j in range(2)]
        for j,c in enumerate(certs): verify_certificate(exact,[row[j] for row in targets],c)
        xor_audits.append({'name':f'z{i}','exact_activations':exact,'certificates':certs,
                           'xor_linearly_separable':[c['separable'] for c in certs]})
    persistent=[j for j in range(4) if all(s['input_bits_linearly_separable'][j] is False for s in interfaces)]
    every_pair=all(any(j in persistent for j in pair) for pair in [(0,1),(2,3)])
    result={'candidate':args.candidate,'arithmetic':'Exact rational evaluation of saved floating-point weights',
            'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'exact_weights':{name:{key:([str(v) for v in value] if isinstance(value[0],Fraction)
                                       else [[str(v) for v in row] for row in value])
                                   for key,value in model.items()} for name,model in weights.items()},
            'input_patterns':inputs,'xor_targets':targets,'interfaces':interfaces,'residual_xor_audits':xor_audits,
            'initial_input_exact':True,'final_xor_exact':True,'final_input_exact':True,
            'xor_logits':[[str(v) for v in row] for row in xor_logits],
            'input_logits':[[str(v) for v in row] for row in input_logits],
            'persistent_nonlinear_input_bits':persistent,'every_xor_has_persistent_nonlinear_input':every_pair,
            'full_result_passes':every_pair and all(v is False for v in xor_audits[0]['xor_linearly_separable'])}
    (OUT/f'candidate-{args.candidate}-exact.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ['candidate','persistent_nonlinear_input_bits',
          'initial_input_exact','final_xor_exact','final_input_exact','full_result_passes']}))
    print('Input readability by block:',[s['input_bits_linearly_separable'] for s in interfaces])


if __name__=='__main__': main()
