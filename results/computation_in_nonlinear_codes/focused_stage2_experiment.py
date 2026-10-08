"""Batched Stage-2 optimization on fixed, exactly reconstructing width-4 codes.

The task-only screen isolates the four width-2 residual blocks from the decoder.
Each batch row is an independent model with its own Adam moments and learning rate.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import minimal_learned_example as base
from width4_codec_experiment import stabilize_code

HERE = Path(__file__).parent
OUT = HERE / "results" / "focused_stage2"


def unpack_transition(parameters):
    layers = []
    for block in range(4):
        p = parameters[:, 12 * block:12 * (block + 1)]
        layers.append((p[:, :4].reshape(-1, 2, 2), p[:, 4:6, None].transpose(1, 2),
                       p[:, 6:10].reshape(-1, 2, 2), p[:, 10:12, None].transpose(1, 2)))
    head_w = parameters[:, 48:52].reshape(-1, 2, 2)
    head_b = parameters[:, 52:54, None].transpose(1, 2)
    return layers, head_w, head_b


def forward(parameters, z, smoothness=0.):
    layers, head_w, head_b = unpack_transition(parameters)
    for a, bias_a, b, bias_b in layers:
        pre = z @ a.transpose(1, 2) + bias_a
        h = F.softplus(pre / smoothness) * smoothness if smoothness else pre.relu()
        z = z + h @ b.transpose(1, 2) + bias_b
    return z @ head_w.transpose(1, 2) + head_b, z


def load_code(seed):
    checkpoint = HERE / "results" / "width4_codec" / f"width-4-seed-{seed}-compressor.pt"
    saved = torch.load(checkpoint, weights_only=True)
    encoder, decoder = base.MLP(4, 2, 4), base.MLP(2, 4, 4)
    encoder.load_state_dict(saved["encoder"])
    decoder.load_state_dict(saved["d0"])
    x = base.truth_table()
    stabilize_code(encoder, decoder, x)
    encoder.requires_grad_(False)
    decoder.requires_grad_(False)
    with torch.no_grad():
        z0 = encoder(x)
        assert base.exact_pattern(decoder(z0), x) == 1.0
    return encoder, decoder, z0


def initialize(z0, branch_scales):
    r = len(z0)
    parameters = torch.empty(r, 54).uniform_(-1/math.sqrt(2), 1/math.sqrt(2))
    layers, _, _ = unpack_transition(parameters)
    z = z0.clone()
    for a, bias_a, b, bias_b in layers:
        projection = z @ a.transpose(1, 2)
        # Place every initial hinge between observed projected inputs.
        ranks = torch.randint(2, 14, (r, 1, 2))
        bias_a.copy_(-projection.sort(dim=1).values.gather(1, ranks))
        b.mul_(branch_scales[:, None, None])
        h = (projection + bias_a).relu()
        update = h @ b.transpose(1, 2)
        bias_b.copy_(-update.mean(dim=1, keepdim=True))
        z = z + update + bias_b
    return parameters


def extract_model(row):
    transition, head = base.Transition(), torch.nn.Linear(2, 2)
    layers, hw, hb = unpack_transition(row[None])
    with torch.no_grad():
        for inside, outside, (a, ba, b, bb) in zip(transition.ins, transition.outs, layers):
            inside.weight.copy_(a[0]); inside.bias.copy_(ba[0, 0])
            outside.weight.copy_(b[0]); outside.bias.copy_(bb[0, 0])
        head.weight.copy_(hw[0]); head.bias.copy_(hb[0, 0])
    return transition, head


def screen_inputs(saved, name="task-only"):
    """Pre-screen candidates; exact certificates are checked after decoder fitting."""
    x = base.truth_table()
    targets = base.xor_targets(x)
    rows = []
    for index in torch.where(saved["best_correct"] == 16)[0].tolist():
        transition, head = extract_model(saved["parameters"][index])
        with torch.no_grad():
            trace = transition.trace(saved["z0"][index])
            task_exact = base.exact_pattern(head(trace[-1][1]), targets) == 1.
            readable = [base.separability_by_column(torch.cat((trace[2*i][1], trace[2*i+1][1]), 1), x)
                        for i in range(4)]
        persistent = [j for j in range(4) if all(not row[j] for row in readable)]
        every_pair = all(any(j in persistent for j in pair) for pair in [(0,1),(2,3)])
        rows.append({"model_index": index, "encoder_seed": [4,9][index//saved["runs_per_code"]],
                     "persistent_nonlinear_inputs_numerical": persistent,
                     "standalone_float32_task_exact": task_exact,
                     "every_pair_numerical": task_exact and every_pair,
                     "unique_final_states": len(torch.unique(trace[-1][1], dim=0))})
    (OUT/f"{name}-input-screen.json").write_text(json.dumps(rows, indent=2)+"\n")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--runs-per-code", type=int, default=64)
    parser.add_argument("--smooth-start", action="store_true")
    args = parser.parse_args()
    assert args.runs_per_code % 16 == 0
    torch.set_num_threads(1)
    torch.manual_seed(90210)
    OUT.mkdir(exist_ok=True)
    codes = [load_code(seed) for seed in [4, 9]]
    runs = 2 * args.runs_per_code
    z0 = torch.stack([c[2] for c in codes]).repeat_interleave(args.runs_per_code, dim=0)
    y = base.xor_targets(base.truth_table())[None].expand(runs, -1, -1)
    lr = torch.tensor([.001, .003, .01, .03]).repeat_interleave(args.runs_per_code // 4).repeat(2)
    scales = torch.tensor([.1, .3, .7, 1.]).repeat(runs // 4)
    parameters = initialize(z0, scales).requires_grad_()
    m, v = torch.zeros_like(parameters), torch.zeros_like(parameters)
    best = parameters.detach().clone()
    best_correct = torch.full((runs,), -1)
    best_loss = torch.full((runs,), float("inf"))
    history = []
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        temperature = .2 * (1-step/(args.steps * .5)) if args.smooth_start and step < args.steps*.5 else 0.
        logits, _ = forward(parameters, z0, temperature)
        loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none").mean((1, 2))
        loss.sum().backward()
        with torch.no_grad():
            gradient = parameters.grad
            m.mul_(.9).add_(gradient, alpha=.1)
            v.mul_(.999).addcmul_(gradient, gradient, value=.001)
            rate = lr * (1 if step < args.steps * .75 else .1)
            parameters.add_(-rate[:, None] * (m / (1-.9**step)) / ((v / (1-.999**step)).sqrt() + 1e-8))
            parameters.grad = None
            if step % 250 == 0 or step == args.steps:
                actual, states = forward(parameters, z0)
                correct = ((actual > 0) == y.bool()).all(-1).sum(-1)
                actual_loss = F.binary_cross_entropy_with_logits(actual, y, reduction="none").mean((1,2))
                improve = (correct > best_correct) | ((correct == best_correct) & (actual_loss < best_loss))
                best[improve] = parameters[improve]
                best_correct[improve] = correct[improve]
                best_loss[improve] = actual_loss[improve]
                if step % 2000 == 0 or step == args.steps:
                    record = {"step": step, "perfect_task_models": int((best_correct == 16).sum()),
                              "best_correct_by_code": [int(x.max()) for x in best_correct.chunk(2)],
                              "elapsed_seconds": round(time.monotonic() - started, 1)}
                    history.append(record)
                    print(json.dumps(record), flush=True)
    logits, z4 = forward(best, z0)
    name = "task-only-smooth" if args.smooth_start else "task-only"
    saved = {"parameters": best, "z0": z0, "logits": logits.detach(), "z4": z4.detach(),
             "best_correct": best_correct, "best_loss": best_loss,
             "encoders": [c[0].state_dict() for c in codes], "decoders": [c[1].state_dict() for c in codes],
             "runs_per_code": args.runs_per_code, "history": history}
    torch.save(saved, OUT / f"{name}.pt")
    screen_inputs(saved, name)
    summary = {"stage": "XOR-only control; reconstruction not in loss", "encoder_seeds": [4,9],
               "runs_per_code": args.runs_per_code, "steps": args.steps,
               "learning_rates": [.001,.003,.01,.03], "branch_scales": [.1,.3,.7,1.],
               "smooth_start": args.smooth_start, "history": history,
               "perfect_task_indices": torch.where(best_correct == 16)[0].tolist(),
               "correct_per_model": best_correct.tolist()}
    (OUT / f"{name}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {name}: {len(summary['perfect_task_indices'])} perfect XOR models", flush=True)


if __name__ == "__main__":
    main()
