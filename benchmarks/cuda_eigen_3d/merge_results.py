"""Combine disjoint completed benchmark batches while retaining provenance."""
import argparse
import csv
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("sources", type=Path, nargs="+")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new output directory")
    metadata, summaries, completed, names = [], [], [], set()
    for source in args.sources:
        info = json.loads((source / "metadata.json").read_text())
        progress = json.loads((source / "progress.json").read_text())
        if progress["status"] != "complete":
            parser.error(f"{source} is not complete")
        cases = {c["name"] for c in info["cases"]}
        if names & cases:
            parser.error(f"duplicate cases: {sorted(names & cases)}")
        names.update(cases)
        metadata.append(info)
        summaries.extend(json.loads((source / "summary.json").read_text()))
        completed.extend(progress["completed"])
    args.output.mkdir(parents=True)
    merged = dict(metadata[0])
    merged.update(cases=[case for info in metadata for case in info["cases"]],
                  measurement_runs=metadata, sources=[str(p.resolve()) for p in args.sources])
    (args.output / "metadata.json").write_text(json.dumps(merged, indent=2))
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2))
    (args.output / "progress.json").write_text(json.dumps(dict(status="complete", active=None,
        total=len(names), completed=completed), indent=2))
    columns = list(dict.fromkeys(key for row in summaries for key in row))
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)
    for source, info in zip(args.sources, metadata):
        for case in info["cases"]:
            folder = source / case["name"]
            if folder.exists():
                shutil.copytree(folder, args.output / case["name"])
    print(f"Merged {len(names)} configurations, {len(summaries)} backend rows.")


if __name__ == "__main__":
    main()
