"""Plot completed suite results without running any solvers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def plot_geometry(output):
    """Draw exact analytic material/PEC/PML slices, not inferred mesh shapes."""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from benchmarks.cuda_eigen_3d.geometries import BUILDERS

    cmap = ListedColormap(["#f7f7f7", "#9bcde8", "#ccbce1", "#333b43"])
    names = {"iris": ("Iris", 5.08), "siw": ("SIW", 0.),
             "perforated_waveguide": ("Perforated guide", 5.33), "dielectric_rod": ("Patched rod", 2.1)}
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    for column, (name, builder) in enumerate(BUILDERS.items()):
        solver, _, _ = builder()
        geom = solver.geometry
        title, y_slice_mm = names[name]
        for row in (0, 1):
            x = np.linspace(*geom.x_span, 350)
            if row == 0:
                other = np.linspace(*geom.z_span, 250)
                xx, zz = np.meshgrid(x, other)
                yy = np.full_like(xx, y_slice_mm*1e-3)
                xlabel, ylabel = "x (mm)", "z (mm), periodic"
                subtitle = f"y={y_slice_mm:g} mm"
            else:
                other = np.linspace(*geom.y_span, 250)
                xx, yy = np.meshgrid(x, other)
                zz = np.full_like(xx, sum(geom.z_span)/2)
                xlabel, ylabel = "x (mm)", "y (mm)"
                subtitle = f"z={sum(geom.z_span)*500:g} mm"
            eps, _ = geom.material_at(xx, yy, zz)
            mask = (eps.real.max(axis=0) > 1.001).astype(int)
            mask[geom.pml_mask(xx, yy)] = 2
            for boundary in geom.boundaries:
                mask[boundary.shape.contains(xx, yy, zz)] = 3
            axis = axes[row, column]
            axis.imshow(mask, origin="lower", interpolation="nearest", cmap=cmap, vmin=0, vmax=3,
                extent=[x[0]*1e3, x[-1]*1e3, other[0]*1e3, other[-1]*1e3], aspect="equal")
            axis.set(xlabel=xlabel, ylabel=ylabel, title=f"{title}: {subtitle}")
            if name == "siw" and row == 1:
                axis.set_yticks([-.8, .8])
    fig.legend(handles=[Patch(facecolor=cmap(i), label=label) for i, label in enumerate(("Air", "Dielectric", "PML", "PEC"))],
               loc="outside lower center", ncol=4)
    fig.suptitle("Periodic benchmark unit cells: analytic geometry slices")
    fig.savefig(output / "geometry_sections.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.loads((args.output / "summary.json").read_text())
    names = {"iris": "Iris-loaded WR-90", "siw": "Via-loaded SIW",
             "perforated_waveguide": "Perforated WR-90", "dielectric_rod": "Patched dielectric rod"}
    backends = {"cpu_eigs": "CPU eigs", "cpu_refined": "CPU refined", "cuda_cudss": "CUDA/cuDSS"}
    colors = {"cpu_eigs": "#3b6b9a", "cpu_refined": "#b66b25", "cuda_cudss": "#24804a"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, (problem, label) in zip(axes.flat, names.items()):
        for backend, legend in backends.items():
            selected = sorted((r for r in rows if r["problem"] == problem and r["backend"] == backend
                and r["threads"] == 1 and r["modes"] == 4 and r["valid"]), key=lambda r: r["field_dofs"])
            if not selected:
                continue
            axis.errorbar([r["field_dofs"] for r in selected], [r["median_seconds"] for r in selected],
                yerr=[[r["median_seconds"]-r["min_seconds"] for r in selected],
                      [r["max_seconds"]-r["median_seconds"] for r in selected]],
                label=legend, marker="o", capsize=3, color=colors[backend])
        axis.set(xscale="log", yscale="log", xlabel="Reduced FEM DOFs (pencil size = 2×)", ylabel="Solve time (seconds)", title=label)
        axis.grid(True, which="both", alpha=.25)
        axis.legend(fontsize=9)
    fig.suptitle("Periodic 3D eigenproblems: CPU vs GPU\nMedian and observed range; one CPU thread, four eigenpairs")
    fig.savefig(args.output / "mesh_timings.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for problem, label in names.items():
        selected = sorted((r for r in rows if r["problem"] == problem and r["backend"] == "cuda_cudss"
            and r["threads"] == 1 and r["modes"] == 4 and r["valid"] and r["speedup_vs_eigs"] is not None),
            key=lambda r: r["field_dofs"])
        axes[0].plot([r["field_dofs"] for r in selected], [r["speedup_vs_eigs"] for r in selected], "o-", label=label)
    axes[0].axhline(1, color="black", linestyle="--", linewidth=1)
    axes[0].set(xscale="log", yscale="log", xlabel="Reduced FEM DOFs (pencil size = 2×)", ylabel="CPU eigs / CUDA time", title="CUDA speedup, one CPU thread")
    axes[0].set_xticks([2000, 5000, 10000, 20000, 30000], labels=["2k", "5k", "10k", "20k", "30k"])
    axes[0].tick_params(axis="x", which="minor", labelbottom=False)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which="both", alpha=.25)
    for problem, label in names.items():
        selected = sorted((r for r in rows if r["problem"] == problem and r["modes"] == 4
            and "-medium" in r["case"] and r["backend"] == "cpu_eigs" and r["valid"]), key=lambda r: r["threads"])
        if selected:
            axes[1].plot([r["threads"] for r in selected], [r["median_seconds"] for r in selected], "o-", label=label)
    axes[1].set_xticks([1, 2, 4, 8, 10, 20])
    axes[1].set(yscale="log", xlabel="Configured BLAS/OpenMP thread limit",
        ylabel="CPU eigs median time (seconds)", title="Thread sensitivity, medium meshes")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="y", alpha=.25)
    fig.savefig(args.output / "speedup_and_threads.png", dpi=180)
    plt.close(fig)
    plot_geometry(args.output)
    write_report(args.output, rows, names)


def hardware_details(output, metadata, records):
    import csv
    import io
    import re
    toolkit = "not captured"
    hardware_path = output / "hardware.json"
    lines = ["Hardware and software", "---------------------", ""]
    fields = {}
    for line in metadata["cpu"].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    runtime = records[0]["runtime"]
    lines.extend([
        f"* CPU: {fields.get('Model name')}; {fields.get('Socket(s)')} sockets, "
        f"{fields.get('Core(s) per socket')} cores/socket, {fields.get('CPU(s)')} logical CPUs, "
        f"{fields.get('Thread(s) per core')} hardware threads/core.",
        f"* CPU maximum reported clock: {fields.get('CPU max MHz')} MHz; L2: {fields.get('L2 cache')}; "
        f"L3: {fields.get('L3 cache')}; NUMA nodes: {fields.get('NUMA node(s)')}.",
        f"* GPU: one {runtime['gpu']}, physical device {metadata['gpu_selection']} "
        "(logical CUDA device 0 in each benchmark worker). No multi-GPU computation.",
    ])
    if hardware_path.exists():
        hardware = json.loads(hardware_path.read_text())
        found = re.search(r"V(\d+\.\d+\.\d+)", hardware["nvcc"]["output"])
        toolkit = found.group(1) if found else "not captured"
        ram = int(hardware['memory']['MemTotal'].split()[0]) / 1024**2
        gpu_rows = list(csv.DictReader(io.StringIO(hardware['gpu']['output']), skipinitialspace=True))
        if gpu_rows:
            gpu = gpu_rows[0]
            lines.extend([
                f"* GPU VRAM: {gpu['memory.total [MiB]']}; compute capability {gpu['compute_cap']}; "
                f"power limit {gpu['power.limit [W]']}; PCI bus {gpu['pci.bus_id']}.",
                f"* NVIDIA driver: {gpu['driver_version']}. Reported maximum SM/memory clocks: "
                f"{gpu['clocks.max.sm [MHz]']} / {gpu['clocks.max.memory [MHz]']} "
                "(supported maxima, not measured sustained clocks).",
            ])
        lines.extend([f"* OS-visible RAM: {ram:.2f} GiB; CPU affinity allowed {len(hardware['affinity'])} logical CPUs. "
                      "Workers were not pinned to a core or NUMA node.",
                      f"* OS: {hardware['platform']}."])
    version = lambda n: f"{n//1000}.{(n%1000)//10}"
    lines.append(f"* CUDA runtime and driver API reported by CuPy: {version(runtime['cuda_runtime'])} / "
                 f"{version(runtime['cuda_driver'])}. Local nvcc toolkit: {toolkit} (full capture in hardware.json).")
    packages = dict(line.split('==', 1) for line in metadata['packages'].splitlines() if '==' in line)
    lines.append(f"* Python: {metadata['python'].split()[0]}; NumPy {runtime['numpy']}; SciPy {runtime['scipy']}; "
                 f"CuPy {runtime['cupy']}; cuDSS package {packages.get('nvidia-cudss-cu13')}; "
                 f"Cython {packages.get('Cython')}; Gmsh {packages.get('gmsh')}.")
    libraries = next((r['runtime']['blas_libraries'] for r in records if r['runtime'].get('blas_libraries')), [])
    for library in libraries:
        lines.append(f"* BLAS ({Path(library['library']).name}): {library.get('config', 'configuration unavailable')}.")
    limits = sorted({r['case']['threads'] for r in records})
    lines.extend([f"* CPU thread limits tested: {', '.join(map(str, limits))}. Set before imports using "
                  "OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS and BLIS_NUM_THREADS. "
                  "The expanded thread sweep also records OpenBLAS runtime thread getters in each result.",
                  "* Arithmetic: complex128 / double precision. CPU eigs uses SciPy ARPACK + SuperLU; "
                  "CPU refined uses the native Cython implementation; GPU refined uses CuPy + cuDSS.", ""])
    return lines


def write_report(output, rows, names):
    metadata = json.loads((output / "metadata.json").read_text())
    cases = {r["case"] for r in rows}
    samples, records = [], []
    for case in cases:
        data = json.loads((output / case / "results.json").read_text())
        samples.extend(data["samples"])
        records.append(data)
    residual = max(s["max_residual"] for s in samples)
    mismatch = max(s["eigenvalue_error"] for s in samples if s["eigenvalue_error"] is not None)
    valid = sum(s["valid"] for s in samples)
    lines = ["Periodic 3D CPU/GPU benchmark report", "="*36, "",
        f"Solver source commit: {metadata['commit']}. Four periodic unit-cell geometries.",
        f"{len(cases)} configurations, {len(rows)} backend/configuration rows, {len(samples)} timed solves.",
        f"Validated samples: {valid}/{len(samples)}. Maximum original-pencil residual: {residual:.3e}.",
        f"Maximum matched relative eigenvalue-magnitude error: {mismatch:.3e}.", "",
        *hardware_details(output, metadata, records),
        "Measurement method", "------------------", "",
        "Timings are medians of three runs; individual values and ranges are retained in summary.csv and per-case results.json.",
        "Mesh assembly and warmup are excluded. Upload, cuDSS analysis/factorization, iteration, CPU extraction and download are included.",
        "These are warmed-process timings: first-use initialization/JIT compilation is represented in the saved warmups, not the reported medians.",
        "CPU eigs uses ARPACK with SuperLU; CPU refined uses the built Cython kernel. CUDA uses CuPy/cuDSS complex FP64.", "",
        "Agreement follows the requested one-to-one eigenvalue magnitude criterion, so symmetry-related forward/backward roots are equally valid.",
        "Complex eigenvalues and their original complex-set mismatch are preserved for inspection. All solves must independently meet the residual target 1e-8.", "",
        "Mesh/DOF scaling", "---------------", ""]

    work = []
    for backend in ("cpu_eigs", "cpu_refined", "cuda_cudss"):
        steps = sorted({s["steps"] for s in samples if s["backend"] == backend})
        restarts = sorted({s["restarts"] for s in samples if s["backend"] == backend and s["restarts"] is not None})
        work.append(f"{backend}: steps {steps}" + (f", restarts {restarts}" if restarts else ""))
    lines[-3:-3] = ["Observed convergence work: " + "; ".join(work) + ".", ""]

    def table(selected):
        lines.extend([".. csv-table::", '   :header: "Case", "Mesh mm", "Threads", "FEM DOFs", "Pencil DOFs", "CPU eigs s", "CPU refined s", "CUDA s", "CUDA speedup"', ""])
        for case in selected:
            grouped = {r["backend"]: r for r in rows if r["case"] == case}
            cpu, refined, gpu = (grouped.get(k) for k in ("cpu_eigs", "cpu_refined", "cuda_cudss"))
            if not all((cpu, refined, gpu)):
                continue
            speed = f"{gpu['speedup_vs_eigs']:.2f}x" if gpu['speedup_vs_eigs'] is not None else "invalid"
            times = [f"{r['median_seconds']:.4f}" if r['median_seconds'] is not None else "failed" for r in (cpu, refined, gpu)]
            cells = [case, str(cpu['size']), str(cpu['threads']), str(cpu['field_dofs']), str(cpu['pencil_dofs']), *times, speed]
            lines.append("   " + ", ".join('"'+c+'"' for c in cells))
        lines.append("")

    table([f"{p}-{level}" for p in names for level in ("coarse", "medium", "fine", "large")])
    lines.extend(["Observed scaling", "~~~~~~~~~~~~~~~~", ""])
    for problem, label in names.items():
        gpu_rows = sorted((r for r in rows if r["problem"] == problem and r["backend"] == "cuda_cudss"
            and r["threads"] == 1 and r["modes"] == 4 and r["valid"]), key=lambda r: r["field_dofs"])
        if len(gpu_rows) > 1:
            first, last = gpu_rows[0], gpu_rows[-1]
            lines.append(f"* {label}: {first['field_dofs']:,} to {last['field_dofs']:,} reduced FEM DOFs; "
                         f"GPU speedup {first['speedup_vs_eigs']:.2f}x to {last['speedup_vs_eigs']:.2f}x relative to one-thread CPU eigs.")
    lines.append("")
    lines.extend([".. image:: mesh_timings.png", "", "CPU thread sensitivity", "----------------------", "",
        "The same medium mesh is compared at each thread limit. These are configured BLAS/OpenMP limits, not claims of parallel ARPACK iteration.",
        "SciPy's sparse LU build links to OpenBLAS. Increasing the limit can have little benefit or increase overhead; the measurements determine the outcome.", ""])
    thread_cases = [r['case'] for r in rows if r['backend']=='cpu_eigs' and r['modes']==4 and '-medium' in r['case']]
    thread_cases.sort(key=lambda c: (next(r['problem'] for r in rows if r['case']==c), next(r['threads'] for r in rows if r['case']==c)))
    table(thread_cases)
    for problem, label in names.items():
        measurements = [r for r in rows if r["problem"] == problem and r["backend"] == "cpu_eigs"
            and r["modes"] == 4 and "-medium" in r["case"] and r["valid"]]
        if measurements:
            baseline = next(r for r in measurements if r["threads"] == 1)
            best = min(measurements, key=lambda r: r["median_seconds"])
            lines.append(f"* {label}: best observed CPU median {best['median_seconds']:.4f} s at thread limit {best['threads']}; "
                         f"{baseline['median_seconds']/best['median_seconds']:.2f}x relative to the one-thread median. "
                         "Small differences may reflect shared-host variability.")
    lines.extend(["", "Thread sweeps use medium meshes; large-mesh speedups are relative to the measured one-thread CPU configuration. "
        "They do not establish the best possible multi-core CPU performance on the large meshes.", ""])
    lines.extend([".. image:: speedup_and_threads.png", "", "Larger candidate sets", "---------------------", "",
        "Medium meshes with 14 eigenpairs and ncv=96, compared with the baseline four eigenpairs/ncv=80. Fourteen candidates correspond to a two-mode 3D FEM request before physical filtering.", ""])
    table([f"{p}-medium-modes14" for p in names])
    geometry_notes = []
    for problem, label in names.items():
        record = next(r for r in records if r["case"]["problem"] == problem)
        info = record["geometry"]
        geometry_notes.append(f"* {label}: {info['frequency_hz']/1e9:g} GHz, period {info['period_mm']:g} mm, "
                              f"target neff {complex(*record['sigma'])}. {info['description']}")
    lines.extend(["Geometry and interpretation", "---------------------------", "", *geometry_notes, "", ".. image:: geometry_sections.png", "",
        "All cases are periodic z unit cells. Filters measure Bloch eigenvalues; these timings do not include finite-device S-parameter calculations.",
        "Eigenvalues are compared before the FEM QEP/Gauss/PML/direction filters. Agreement between backends at each density is distinct from convergence of the physical model under mesh/PML refinement.",
        "Other users had active jobs on this shared host. Measurements use GPU 3 and sequential workloads; host-load/GPU snapshots are saved in metadata.json.", "",
        "Reproduction and files", "----------------------", "",
        "Source and methodology: benchmarks/cuda_eigen_3d/README.rst, geometries.py, run_benchmarks.py.",
        "summary.csv and summary.json contain all backend/configuration rows. Each case directory retains the timed samples, complex eigenvalues, residuals, iteration counts and mesh statistics.",
        "metadata.json records commands, package versions, hardware, thread settings, source hashes and measurement-batch provenance.",
        "raw/ preserves the three original measurement batches and logs; hardware.json holds the dedicated machine-specification capture; audit.json records final integrity checks.", "",
        "Threading references: https://docs.scipy.org/doc/scipy/tutorial/parallel_execution.html and https://www.arpack.org/installation .", ""])
    (output / "REPORT.rst").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
