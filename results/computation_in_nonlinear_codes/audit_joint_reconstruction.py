"""Exactly evaluate every sweep checkpoint and certify all successful XOR circuits.

LPs propose witnesses; rational arithmetic verifies every reported readability
classification. Unresolved cases are explicitly excluded from the 0--4 histogram.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction as Q
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import sympy as sp
from sympy.solvers.simplex import InfeasibleLPError, linprog as exact_lp

from exact_focused_stage2_audit import linear, mlp, predictions, relu
from verify_width4_codec_experiment import verify_certificate
from width4_codec_experiment import certify

HERE = Path(__file__).resolve().parent


def rational(value):
    return [rational(v) for v in value] if isinstance(value, list) else Q(value)


def serialize(value):
    if isinstance(value, Q):
        return str(value)
    if isinstance(value, list):
        return [serialize(v) for v in value]
    if isinstance(value, dict):
        return {k: serialize(v) for k, v in value.items()}
    return value


def verified_certificate(rows, labels):
    """Use only certificates checked against the exact, original coordinates."""
    exact = serialize(rows)
    numerical = np.array([[float(v) for v in row] for row in rows])
    try:
        result = certify(numerical, labels, exact_values=exact)
        if result['separable'] is not None:
            verify_certificate(exact, labels, result)
            return result
    except (AssertionError, ValueError):
        pass
    # Remove exactly redundant affine coordinates before retrying: float roundoff
    # must not manufacture a new direction in an always-active ReLU branch.
    differences = sp.Matrix([[sp.Rational(v - c) for v, c in zip(row, rows[0])] for row in rows[1:]])
    _, columns = differences.rref()
    reduced = [[row[j] for j in columns] for row in rows]
    result = None
    if columns:
        try:
            result = certify(np.array([[float(v) for v in row] for row in reduced]), labels,
                             exact_values=serialize(reduced))
        except (AssertionError, ValueError):
            pass
    if result is None or result['separable'] is None:
        # Exact feasibility of a unit-margin affine separator. This is a fallback
        # for numerical degeneracies, not a reinterpretation of solver failure.
        design = sp.Matrix([[-(2 * label - 1) * sp.Rational(v) for v in [*row, Q(1)]]
                            for row, label in zip(reduced, labels)])
        try:
            _, solution = exact_lp([0] * design.cols, design, [-1] * len(rows), bounds=(None, None))
            coefficients = [Q(str(v)) for v in solution]
            margins = [(2 * t - 1) * sum(a * b for a, b in zip(coefficients, [*row, Q(1)]))
                       for row, t in zip(reduced, labels)]
            assert min(margins) > 0
            result = {'separable': True, 'coefficients': list(map(str, coefficients)),
                      'minimum_exact_margin': str(min(margins))}
        except InfeasibleLPError:
            columns_hull = [[(1 if labels[i] else -1) * sp.Rational(v) for v in row]
                            + [int(labels[i]), int(not labels[i])] for i, row in enumerate(reduced)]
            matrix = sp.Matrix.hstack(*(sp.Matrix(col) for col in columns_hull))
            target = sp.Matrix([0] * len(columns) + [1, 1])
            _, solution = exact_lp([0] * len(rows), matrix.col_join(-matrix),
                                   list(target) + list(-target))
            result = {'separable': False, 'exact_hull_support': [
                {'input_index': i, 'label': int(labels[i]), 'weight': str(v)}
                for i, v in enumerate(solution) if v > 0]}
    if result['separable']:
        expanded = ["0"] * len(rows[0]) + [result['coefficients'][-1]]
        for j, value in zip(columns, result['coefficients'][:-1]):
            expanded[j] = value
        result['coefficients'] = expanded
    assert result['separable'] is not None
    verify_certificate(exact, labels, result)
    return result


def evaluate(weights, inputs):
    z = mlp(inputs, weights['encoder'])
    states, interfaces = [z], []
    w = weights['transition']
    for k in range(4):
        h = relu(linear(z, w[f'ins.{k}.weight'], w[f'ins.{k}.bias']))
        interfaces.append([a + b for a, b in zip(z, h)])
        update = linear(h, w[f'outs.{k}.weight'], w[f'outs.{k}.bias'])
        z = [[a + b for a, b in zip(row, change)] for row, change in zip(z, update)]
        states.append(z)
    return states, interfaces, linear(z, weights['head']['weight'], weights['head']['bias']), mlp(z, weights['d4'])


def initial_audit(weights, inputs, targets):
    z = mlp(inputs, weights['encoder'])
    forced = [a + b - c for a, b, c in zip(z[8], z[4], z[0])]
    discrepancy = [a - b for a, b in zip(z[12], forced)]
    return {
        'initial_reconstruction_correct': predictions(mlp(z, weights['d0'])) == inputs,
        'encoder_non_affine': any(discrepancy),
        'affine_witness': {'input': inputs[12], 'forced': serialize(forced),
                           'actual': serialize(z[12]), 'difference': serialize(discrepancy)},
        'initial_xor_certificates': [verified_certificate(z, [row[j] for row in targets]) for j in range(2)],
        'initial_input_certificates': [verified_certificate(z, [row[j] for row in inputs]) for j in range(4)],
    }


def summarize(rows):
    xor = [r for r in rows if r['xor_correct_patterns'] == 16]
    both = [r for r in xor if r['reconstruction_correct_patterns'] == 16]
    def distribution(items):
        counts = Counter(r['persistent_unreadable_count'] for r in items)
        return {str(k): counts[k] for k in range(5)}
    return {'runs': len(rows), 'xor_correct': len(xor),
            'reconstruction_correct': sum(r['reconstruction_correct_patterns'] == 16 for r in rows),
            'both_correct': len(both),
            'distinct_final_codes': sum(r['distinct_final_codes'] == 16 for r in rows),
            'unreadable_bit_histogram_among_xor_correct': distribution(xor),
            'unreadable_bit_histogram_among_both_correct': distribution(both),
            'one_persistent_operand_per_xor_among_xor_correct': sum(r['every_xor_has_persistent_nonlinear_input'] for r in xor),
            'one_persistent_operand_per_xor_among_both_correct': sum(r['every_xor_has_persistent_nonlinear_input'] for r in both),
            'six_claims_pass': sum(r['six_claims_pass'] for r in rows),
            'all_requirements_pass': sum(r['all_requirements_pass'] for r in rows),
            'unresolved_xor_successes': sum(r['persistent_unreadable_count'] is None for r in xor)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=HERE / 'results' / 'joint_decoder_refinement')
    args = parser.parse_args()
    source = args.folder / 'trained-weights.json'
    saved = json.loads(source.read_text())
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    inputs = list(map(list, itertools.product((0, 1), repeat=4)))
    targets = [[a ^ b, c ^ d] for a, b, c, d in inputs]
    encoders = {seed: {name: {k: rational(v) for k, v in w.items()} for name, w in code.items()}
                for seed, code in saved['encoders'].items()}
    initial = {seed: initial_audit(code, inputs, targets) for seed, code in encoders.items()}
    for seed, audit in initial.items():
        assert audit['initial_reconstruction_correct'] and audit['encoder_non_affine']
        assert all(c['separable'] is False for c in audit['initial_xor_certificates'])
    (args.folder / 'encoder-audits.json').write_text(json.dumps(initial, indent=2) + '\n')
    audit_folder = args.folder / 'audits'
    audit_folder.mkdir(exist_ok=True)
    rows, eligible = [], []
    started = time.monotonic()
    for run in saved['runs']:
        seed = str(run['encoder_seed'])
        weights = {**encoders[seed], **{name: {k: rational(v) for k, v in run[name].items()}
                                      for name in ['transition', 'head', 'd4']}}
        states, live, logits, reconstruction = evaluate(weights, inputs)
        record = {k: run[k] for k in ['id', 'matched_run', 'encoder_seed', 'lambda', 'learning_rate',
                                     'branch_scale', 'decoder_initialization', 'checkpoint_step', 'selection_loss']}
        for key in ('source_circuit', 'decoder_restart', 'circuit_rate_multiplier'):
            if key in run:
                record[key] = run[key]
        record.update(xor_correct_patterns=sum(a == b for a, b in zip(predictions(logits), targets)),
                      reconstruction_correct_patterns=sum(a == b for a, b in zip(predictions(reconstruction), inputs)),
                      distinct_final_codes=len(set(map(tuple, states[-1]))),
                      persistent_unreadable_bits=None, persistent_unreadable_count=None,
                      every_xor_has_persistent_nonlinear_input=False, six_claims_pass=False,
                      all_requirements_pass=False)
        if record['xor_correct_patterns'] == 16:
            audit_file = audit_folder / f"run-{run['id']:03d}.json"
            if audit_file.exists():
                certificates = json.loads(audit_file.read_text())
                assert certificates['weights_source_sha256'] == source_hash
                certs = certificates['interface_certificates']
                for rows_at_interface, block in zip(live, certs):
                    for j, certificate in enumerate(block):
                        assert certificate['separable'] is not None
                        verify_certificate(rows_at_interface, [row[j] for row in inputs], certificate)
            else:
                certs = [[verified_certificate(rows_at_interface, [row[j] for row in inputs])
                          for j in range(4)] for rows_at_interface in live]
                audit_file.write_text(json.dumps({'id': run['id'], 'weights_source_sha256': source_hash,
                                                 'interface_certificates': certs}, indent=2) + '\n')
            persistent = [j for j in range(4) if all(c[j]['separable'] is False for c in certs)]
            # Projecting a joint interface to either adjacent residual state is
            # affine, so the same hull witnesses also certify those residuals.
            for k, block in enumerate(certs):
                for j in persistent:
                    for state in [states[k], states[k+1]]:
                        verify_certificate(state, [row[j] for row in inputs], block[j])
            every_pair = all(any(j in persistent for j in pair) for pair in [(0, 1), (2, 3)])
            record.update(persistent_unreadable_bits=persistent, persistent_unreadable_count=len(persistent),
                          every_xor_has_persistent_nonlinear_input=every_pair,
                          six_claims_pass=every_pair,
                          all_requirements_pass=every_pair and record['reconstruction_correct_patterns'] == 16)
            if record['all_requirements_pass'] and run['lambda'] > 0:
                eligible.append((record, weights, states, live, logits, reconstruction, certs))
        rows.append(record)
        if len(rows) % 16 == 0:
            print(json.dumps({'audited': len(rows), 'elapsed_seconds': round(time.monotonic()-started, 1),
                              'xor_correct': sum(r['xor_correct_patterns'] == 16 for r in rows),
                              'all_requirements': sum(r['all_requirements_pass'] for r in rows)}), flush=True)
    arms = []
    for weight in saved['protocol']['reconstruction_weights']:
        arm_rows = [r for r in rows if r['lambda'] == weight]
        arms.append({'lambda': weight, **summarize(arm_rows),
                     'by_encoder': {seed: summarize([r for r in arm_rows if str(r['encoder_seed']) == seed])
                                    for seed in encoders}})
        if arm_rows and 'source_circuit' in arm_rows[0]:
            arms[-1]['by_source_circuit'] = {
                str(source): summarize([r for r in arm_rows if r['source_circuit'] == source])
                for source in sorted({r['source_circuit'] for r in arm_rows})}
    result = {'arithmetic': 'Exact rational evaluation of saved floating-point weights and verified certificates',
              'weights_source_sha256': source_hash, 'protocol': saved['protocol'],
              'arms': arms, 'runs': rows, 'selected_example': None,
              'example_selection_rule': 'Prefer lambda=1; among eligible runs choose smallest joint selection loss, then id'}
    if eligible:
        record, weights, states, live, logits, reconstruction, certs = min(
            eligible, key=lambda item: (item[0]['lambda'] != 1., item[0]['selection_loss'], item[0]['id']))
        xor_audits = []
        for k, state in enumerate(states):
            c = [verified_certificate(state, [row[j] for row in targets]) for j in range(2)]
            xor_audits.append({'name': f'z{k}', 'exact_activations': serialize(state),
                               'certificates': c, 'xor_linearly_separable': [v['separable'] for v in c]})
        selected = {'candidate': record['id'], 'training': record, 'arithmetic': result['arithmetic'],
                    'weights_source_sha256': source_hash, 'exact_weights': serialize(weights),
                    'input_patterns': inputs, 'xor_targets': targets,
                    'interfaces': [{'name': f'z{k}+r{k+1}', 'exact_activations': serialize(values),
                                    'certificates': c, 'input_bits_linearly_separable': [v['separable'] for v in c]}
                                   for k, (values, c) in enumerate(zip(live, certs))],
                    'residual_xor_audits': xor_audits, 'initial_input_exact': True,
                    'final_xor_exact': True, 'final_input_exact': True,
                    'xor_logits': serialize(logits), 'input_logits': serialize(reconstruction),
                    'persistent_nonlinear_input_bits': record['persistent_unreadable_bits'],
                    'every_xor_has_persistent_nonlinear_input': True, 'full_result_passes': True}
        (args.folder / 'selected-example-exact.json').write_text(json.dumps(selected, indent=2) + '\n')
        result['selected_example'] = record
    (args.folder / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'arms': arms, 'selected_example': result['selected_example']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
