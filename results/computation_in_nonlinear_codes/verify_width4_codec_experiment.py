"""Verify saved predictions and rational certificates without training or numerical LPs."""
from fractions import Fraction
import itertools
import json
from pathlib import Path


def verify_certificate(values, labels, certificate):
    rows = [[Fraction(v) for v in row] for row in values]
    if certificate["separable"] is None:
        assert certificate.get("reason")
        return
    if certificate["separable"]:
        weights = list(map(Fraction, certificate["coefficients"]))
        assert len(weights) == len(rows[0]) + 1
        margins = [(2 * label - 1) * sum(w * v for w, v in zip(weights, row + [Fraction(1)]))
                   for row, label in zip(rows, labels)]
        assert min(margins) > 0
        assert min(margins) == Fraction(certificate["minimum_exact_margin"])
    else:
        sums = [Fraction(0), Fraction(0)]
        points = [[Fraction(0)] * len(rows[0]) for _ in range(2)]
        for item in certificate["exact_hull_support"]:
            i, label, weight = item["input_index"], item["label"], Fraction(item["weight"])
            assert label == labels[i] and weight > 0
            sums[label] += weight
            points[label] = [a + weight * b for a, b in zip(points[label], rows[i])]
        assert sums == [1, 1]
        assert points[0] == points[1]


def accuracy(logits, targets):
    return sum([int(v > 0) for v in row] == target for row, target in zip(logits, targets)) / len(targets)


def main():
    folder = Path(__file__).parent / "results" / "width4_codec"
    count = 0
    for path in sorted(folder.glob("*.json")):
        if "compressor" in path.name:
            continue
        r = json.loads(path.read_text())
        inputs = r["input_patterns"]
        assert inputs == list(map(list, itertools.product((0, 1), repeat=4)))
        targets = [[a ^ b, c ^ d] for a, b, c, d in inputs]
        assert r["xor_targets"] == targets
        assert accuracy(r["initial_input_logits"], inputs) == r["initial_input_exact"]
        assert accuracy(r["input_logits"], inputs) == r["final_input_exact"]
        assert accuracy(r["task_logits"], targets) == r["final_task_exact"]
        for interface in r["interfaces"]:
            for j, certificate in enumerate(interface["certificates"]):
                verify_certificate(interface["activations"], [row[j] for row in inputs], certificate)
            assert interface["input_bits_linearly_separable"] == [c["separable"] for c in interface["certificates"]]
        for state in r["residual_target_audits"]:
            for j, certificate in enumerate(state["certificates"]):
                verify_certificate(state["activations"], [row[j] for row in targets], certificate)
            assert state["xor_linearly_separable"] == [c["separable"] for c in state["certificates"]]
        persistent = [j for j in range(4) if all(row["input_bits_linearly_separable"][j] is False
                                               for row in r["interfaces"])]
        assert persistent == r["persistent_nonlinear_input_bits"]
        every_pair = all(any(j in persistent for j in pair) for pair in [(0, 1), (2, 3)])
        assert every_pair == r["every_xor_has_persistent_nonlinear_input"]
        expected_pass = (r["initial_input_exact"] == r["final_task_exact"] == r["final_input_exact"] == 1
                         and all(v is False for v in r["initial_xor_linearly_separable"]) and every_pair)
        assert expected_pass == r["full_result_passes"]
        unresolved = sum(c["separable"] is None for block in r["interfaces"] for c in block["certificates"])
        print(f"PASS {path.name}: predictions and certificates; {unresolved} input audits explicitly unresolved")
        count += 1
    assert count


if __name__ == "__main__":
    main()
