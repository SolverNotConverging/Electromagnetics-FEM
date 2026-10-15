"""Validate maintained RST documents and relative reference links."""

import io
from pathlib import Path
import re
from docutils.core import publish_doctree
from docutils import nodes

ROOT = Path(__file__).resolve().parents[1]


def main():
    errors = []
    files = []
    for directory in (
        "fem_common",
        "fem_adaptivity",
        "fem_electrostatics",
        "fem_periodic_modes",
        "fem_waveguide_modes",
        "fem_waveguide_scattering",
        "apps",
        "docs",
        "tests",
        "benchmarks",
    ):
        files.extend((ROOT / directory).rglob("*.rst"))
    for path in files:
        if any(part in ("build", "dist") or part.endswith(".egg-info") for part in path.parts):
            continue
        source = path.read_text(encoding="utf-8")
        warnings = io.StringIO()
        document = publish_doctree(
            source,
            source_path=str(path),
            settings_overrides={
                "warning_stream": warnings,
                "halt_level": 6,
                "report_level": 2,
                "syntax_highlight": "none",
            },
        )
        if warnings.getvalue():
            errors.append(warnings.getvalue())
        links = [node["refuri"] for node in document.findall(nodes.reference) if "refuri" in node]
        links += [node["uri"] for node in document.findall(nodes.image)]
        for link in links:
            if "://" in link or link.startswith(("mailto:", "#")):
                continue
            if not (path.parent / link.split("#")[0]).exists():
                errors.append(f"{path}: broken local link {link}")
    markdown = [
        *ROOT.glob("*.md"),
        *(ROOT / "docs").rglob("*.md"),
        *(ROOT / "benchmarks").rglob("*.md"),
        *(ROOT / "apps").rglob("*.md"),
    ]
    for path in markdown:
        for link in re.findall(r"\[[^\]\n]+\]\(([^)\n]+)\)", path.read_text(encoding="utf-8")):
            if "://" in link or link.startswith(("mailto:", "#")):
                continue
            if not (path.parent / link.split("#")[0]).exists():
                errors.append(f"{path}: broken local link {link}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Validated {len(files)} maintained RST documents and relative links.")


if __name__ == "__main__":
    main()
