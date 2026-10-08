"""Shared model definitions and training routines for the computation example.

Run directly, it trains one small example; the README's results come from the
three-stage pipeline in REPRODUCING.md.  In that small example, a
reconstruction-only 4 -> 2 compressor is trained first and frozen.  A width-2
residual ReLU circuit then learns two XOR relations while an adapted nonlinear
decoder preserves all four source bits.
"""

from __future__ import annotations

import copy
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linprog
from sklearn.linear_model import LinearRegression


torch.set_num_threads(1)

N_SOURCES = 4
CODE_WIDTH = 2
DEPTH = 4
SEED = 1


def truth_table() -> torch.Tensor:
    return torch.tensor(list(itertools.product((0.0, 1.0), repeat=N_SOURCES)))


def xor_targets(x: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (((x[:, 0] > 0.5) ^ (x[:, 1] > 0.5)).float(),
         ((x[:, 2] > 0.5) ^ (x[:, 3] > 0.5)).float()),
        dim=1,
    )


class MLP(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden_width: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Transition(nn.Module):
    def __init__(self):
        super().__init__()
        self.ins = nn.ModuleList(nn.Linear(CODE_WIDTH, CODE_WIDTH) for _ in range(DEPTH))
        self.outs = nn.ModuleList(nn.Linear(CODE_WIDTH, CODE_WIDTH) for _ in range(DEPTH))

    def trace(self, z: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        states = [("z0", z)]
        for index, (inside, outside) in enumerate(zip(self.ins, self.outs), 1):
            h = torch.relu(inside(z))
            states.append((f"h{index}", h))
            z = z + outside(h)
            states.append((f"z{index}", z))
        return states

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.trace(z)[-1][1]


def exact_pattern(logits: torch.Tensor, labels: torch.Tensor) -> float:
    prediction = (logits > 0).float()
    return float(torch.all(prediction == labels, dim=1).float().mean())


def affine_fvu(x: torch.Tensor, z: torch.Tensor) -> float:
    x_np, z_np = x.numpy(), z.numpy()
    prediction = LinearRegression().fit(x_np, z_np).predict(x_np)
    residual = np.square(z_np - prediction).sum()
    centered = np.square(z_np - z_np.mean(axis=0, keepdims=True)).sum()
    return float(residual / max(centered, 1e-12))


def linearly_separable(z: torch.Tensor, labels: torch.Tensor) -> bool:
    """Exact affine-separability certificate for a finite binary dataset."""
    z_np = z.detach().numpy()
    signs = 2.0 * labels.detach().numpy() - 1.0
    design = np.concatenate((z_np, np.ones((len(z_np), 1))), axis=1)
    result = linprog(
        np.zeros(design.shape[1]),
        A_ub=-(signs[:, None] * design),
        b_ub=-np.ones(len(z_np)),
        bounds=[(None, None)] * design.shape[1],
        method="highs",
    )
    return bool(result.success)


def separability_by_column(z: torch.Tensor, labels: torch.Tensor) -> list[bool]:
    return [linearly_separable(z, labels[:, index]) for index in range(labels.shape[1])]


def train_compressor(x: torch.Tensor, hidden_width: int = 16) -> tuple[MLP, MLP, float]:
    candidates = []
    for restart in range(3):
        torch.manual_seed(SEED + 10_000 * restart)
        encoder = MLP(N_SOURCES, CODE_WIDTH, hidden_width)
        decoder = MLP(CODE_WIDTH, N_SOURCES, hidden_width)
        optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=3e-3)
        for _ in range(5_000):
            loss = nn.functional.binary_cross_entropy_with_logits(decoder(encoder(x)), x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            loss = nn.functional.binary_cross_entropy_with_logits(decoder(encoder(x)), x).item()
        candidates.append((loss, encoder, decoder))

    loss, encoder, decoder = min(candidates, key=lambda item: item[0])
    encoder.eval()
    decoder.eval()
    for parameter in [*encoder.parameters(), *decoder.parameters()]:
        parameter.requires_grad_(False)
    return encoder, decoder, loss


def train_computation(
    x: torch.Tensor,
    y: torch.Tensor,
    z0: torch.Tensor,
    original_decoder: MLP,
    progress=None,
) -> tuple[Transition, nn.Linear, MLP]:
    candidates = []
    for restart in range(10):
        torch.manual_seed(100_000 + SEED + 10_000 * restart + 100 * DEPTH)
        transition = Transition()
        head = nn.Linear(CODE_WIDTH, 2)
        decoder = copy.deepcopy(original_decoder)
        for parameter in decoder.parameters():
            parameter.requires_grad_(True)
        parameters = [*transition.parameters(), *head.parameters(), *decoder.parameters()]
        optimizer = torch.optim.Adam(parameters, lr=3e-3)
        for _ in range(10_000):
            z5 = transition(z0)
            task_loss = nn.functional.binary_cross_entropy_with_logits(head(z5), y)
            reconstruction_loss = nn.functional.binary_cross_entropy_with_logits(decoder(z5), x)
            loss = task_loss + 0.1 * reconstruction_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            z5 = transition(z0)
            task_loss = nn.functional.binary_cross_entropy_with_logits(head(z5), y).item()
            reconstruction_loss = nn.functional.binary_cross_entropy_with_logits(decoder(z5), x).item()
            task_exact = exact_pattern(head(z5), y)
            source_exact = exact_pattern(decoder(z5), x)
        selection = (-(task_exact + source_exact), task_loss + 0.1 * reconstruction_loss)
        candidates.append((selection, transition, head, decoder))
        if progress is not None:
            progress({"restart": restart, "task_exact": task_exact,
                      "input_exact": source_exact, "loss": selection[1]})

    _, transition, head, decoder = min(candidates, key=lambda item: item[0])
    transition.eval()
    head.eval()
    decoder.eval()
    return transition, head, decoder


def run() -> dict:
    x = truth_table()
    y = xor_targets(x)
    encoder, original_decoder, compressor_loss = train_compressor(x)

    with torch.no_grad():
        z0 = encoder(x)
    transition, head, final_decoder = train_computation(x, y, z0, original_decoder)

    with torch.no_grad():
        trace = [(name, state.detach()) for name, state in transition.trace(z0)]
        z5 = trace[-1][1]
        task_prediction = (head(z5) > 0).int()
        source_prediction = (final_decoder(z5) > 0).int()

    residual_states = [trace[2 * index][1] for index in range(DEPTH + 1)]
    live_interfaces = []
    for index in range(DEPTH):
        live = torch.cat((trace[2 * index][1], trace[2 * index + 1][1]), dim=1)
        source_separable = separability_by_column(live, x)
        live_interfaces.append({
            "name": f"z{index}+h{index + 1}",
            "source_bits_linearly_separable": source_separable,
            "complete_linear_unpack_exists": all(source_separable),
        })

    rows = []
    for index in range(len(x)):
        rows.append({
            "x": x[index].int().tolist(),
            "xor": y[index].int().tolist(),
            "z0": [round(float(value), 4) for value in z0[index]],
            "z5": [round(float(value), 4) for value in z5[index]],
            "xor_prediction": task_prediction[index].tolist(),
            "source_prediction": source_prediction[index].tolist(),
        })

    result = {
        "status": "single_seed_exhaustive_existence_example",
        "architecture": {
            "sources": N_SOURCES,
            "code_width": CODE_WIDTH,
            "xor_targets": 2,
            "residual_relu_blocks": DEPTH,
            "branch_width": CODE_WIDTH,
            "compressor_training": "reconstruction only, then frozen",
            "final_decoder": "adapted after compression",
        },
        "selection_disclosure": {
            "compressor_seed": SEED,
            "compressor_restarts": 3,
            "task_restarts": 10,
            "evaluated_patterns": 16,
            "note": "Seed 1 was retained after a small post-hoc seed search because most 4->2 compressors were nearly affine.",
        },
        "compressor_loss": compressor_loss,
        "initial_source_exact_pattern": exact_pattern(original_decoder(z0), x),
        "initial_affine_fvu": affine_fvu(x, z0),
        "initial_xor_linearly_separable": separability_by_column(z0, y),
        "final_xor_exact_pattern": exact_pattern(head(z5), y),
        "final_xor_linearly_separable": separability_by_column(z5, y),
        "final_linear_head": {
            "weight": [[round(float(value), 6) for value in row] for row in head.weight.detach()],
            "bias": [round(float(value), 6) for value in head.bias.detach()],
        },
        "final_source_exact_pattern": exact_pattern(final_decoder(z5), x),
        "residual_affine_fvu": [affine_fvu(x, state) for state in residual_states],
        "live_interfaces": live_interfaces,
        "truth_table_trace": rows,
    }
    result["passes_declared_scope"] = bool(
        result["initial_source_exact_pattern"] == 1.0
        and result["initial_affine_fvu"] > 0.1
        and not any(result["initial_xor_linearly_separable"])
        and result["final_xor_exact_pattern"] == 1.0
        and all(result["final_xor_linearly_separable"])
        and result["final_source_exact_pattern"] == 1.0
        and min(result["residual_affine_fvu"]) > 0.1
        and not any(item["complete_linear_unpack_exists"] for item in live_interfaces)
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    rendered = json.dumps(run(), indent=2) + "\n"
    if arguments.output:
        arguments.output.write_text(rendered)
    else:
        print(rendered, end="")
