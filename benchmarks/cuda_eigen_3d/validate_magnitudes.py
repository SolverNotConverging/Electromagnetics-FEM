"""Revalidate saved timings using the user's eigenvalue-magnitude criterion.

Original measurements are preserved. No solves are run and no timings change.
Complex eigenvalues and complex-set mismatch remain available as diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def magnitude_error(reference, candidate):
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if reference.size != candidate.size or reference.size == 0:
        return None
    if not (np.isfinite(reference).all() and np.isfinite(candidate).all()):
        return None
    distances = abs(abs(reference)[:, None] - abs(candidate)[None, :])
    left, right = linear_sum_assignment(distances)
    return float(np.max(distances[left, right] / np.maximum(1., abs(reference[left]))))


def eigenvalues(sample):
    return np.array([complex(*x) for x in sample["eigenvalues"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("output must differ from source to preserve raw measurements")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.source / "metadata.json").read_text())
    metadata["validation"] = dict(source=str(args.source.resolve()),
        criterion="One-to-one matching of eigenvalue magnitudes, as requested by the user; max abs(|lambda_ref|-|lambda_test|)/max(1,|lambda_ref|) < 1e-5. Every sample must also converge and meet its original-pencil residual tolerance. Complex eigenvalues are retained; complex-set equality is diagnostic only.")
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    summaries = []
    for case in metadata["cases"]:
        source_file = args.source / case["name"] / "results.json"
        if not source_file.exists():
            continue
        data = json.loads(source_file.read_text())
        if data.get("status") != "complete":
            # Partial/failed cases remain visible in the copied progress file.
            continue
        reference_sample = next((s for s in data["warmups"] if s["backend"] == "cpu_eigs"), None)
        reference = eigenvalues(reference_sample) if reference_sample else np.array([])
        reference_good = bool(reference_sample and reference_sample["converged"]
            and reference_sample["max_residual"] <= case["tol"]
            and reference.size == case["modes"] and np.isfinite(reference).all())
        for sample in data["warmups"] + data["samples"]:
            mismatch = magnitude_error(reference, eigenvalues(sample))
            sample["complex_eigenvalue_error"] = sample.get("complex_eigenvalue_error", sample["eigenvalue_error"])
            sample["complex_set_match"] = bool(sample["complex_eigenvalue_error"] is not None and sample["complex_eigenvalue_error"] < 1e-5)
            sample["magnitude_eigenvalue_error"] = mismatch
            sample["eigenvalue_error"] = mismatch
            sample["valid"] = bool(reference_good and sample["converged"]
                and sample["max_residual"] <= case["tol"]
                and mismatch is not None and mismatch < 1e-5)
        for row in data["summaries"]:
            samples = [s for s in data["samples"] if s["backend"] == row["backend"]]
            row["valid"] = bool(len(samples) == metadata["repeats"] and all(s["valid"] for s in samples)
                                and row["backend"] not in data["errors"])
            row["complex_set_match"] = bool(samples and all(s["complex_set_match"] for s in samples))
            row["max_complex_eigenvalue_error"] = row.get("max_complex_eigenvalue_error", row["max_eigenvalue_error"])
            row["max_eigenvalue_error"] = max((s["magnitude_eigenvalue_error"] for s in samples
                                               if s["magnitude_eigenvalue_error"] is not None), default=None)
        cpu = next(r for r in data["summaries"] if r["backend"] == "cpu_eigs")
        for row in data["summaries"]:
            row["speedup_vs_eigs"] = cpu["median_seconds"] / row["median_seconds"] if cpu["valid"] and row["valid"] else None
            summaries.append(dict(case=case["name"], problem=case["problem"], size=case["size"],
                modes=case["modes"], ncv=case["ncv"], tol=case["tol"], threads=case["threads"],
                pencil_dofs=data["pencil_dofs"], field_dofs=data["field_dofs"], **row))
        data["validation"] = metadata["validation"]
        folder = args.output / case["name"]
        folder.mkdir(exist_ok=True)
        (folder / "results.json").write_text(json.dumps(data, indent=2))
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2))
    if summaries:
        with (args.output / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(summaries)
    progress = json.loads((args.source / "progress.json").read_text())
    (args.output / "progress.json").write_text(json.dumps(progress, indent=2))
    invalid = [r for r in summaries if not r["valid"]]
    print(f"Validated {len(summaries)} backend/case rows; {len(invalid)} invalid.")
    for row in invalid:
        print(row["case"], row["backend"], row["max_residual"], row["max_eigenvalue_error"], row["error"])
    if invalid or len(summaries) != 3*len(metadata["cases"]) or progress["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
