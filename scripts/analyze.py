#!/usr/bin/env python3
"""Tables and figures for the Assignment 3 report from one benchmark run.

Usage: python3 scripts/analyze.py [results/run_XXXX]   (default: latest run)

Statistics per configuration use the MEASURED repetitions only (warm-ups stay in
the raw CSV but are excluded): median, Q1, Q3, IQR. All times are per
multiplication (one repetition times `batch` multiplications).

Derived values (same n, medians of the measured repetitions)
  S(p)      speedup     = median T_seq(n) / median T_p(n)
  E(p)      efficiency  = S(p) / p
  S range   [Q1_seq / Q3_p, Q3_seq / Q1_p]: a conservative band that combines the
            spread of both measurements (not a confidence interval)
  overhead  T_p=1(n) / T_seq(n) - 1: cost of going through the thread pool with a
            single worker
  CPU util. process CPU time / wall time during a timed repetition (1.0 = one
            core busy all the time; includes JVM background threads)
  GFLOP/s   2 n^3 / median
Vector extension: speedup of each variant against `novec` and against `auto`
with the same n and p.

Outputs: report/tables/*.csv|*.md and report/figures/*.png
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Reference palette (validated with the dataviz validator, light mode).
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# Sequential blue ramp (light -> dark) for the ordered sizes n.
RAMP = ["#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
METHOD_STYLE = {"static": (BLUE, "o", "-"), "stream": (ORANGE, "s", "--"), "seq": (INK2, "D", ":")}
METHOD_LABEL = {"seq": "Sequential i-k-j (A2)", "static": "Static row blocks (thread pool)",
                "stream": "Parallel stream (ForkJoin)"}
VARIANT_STYLE = {"novec": (VIOLET, "No auto-vectorisation"), "auto": (BLUE, "Default JIT (auto)"),
                 "vapi": (AQUA, "Explicit SIMD (Vector API)")}


def latest_run():
    runs = sorted((ROOT / "results").glob("run_*"))
    if not runs:
        sys.exit("No runs found in results/")
    return runs[-1]


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_table(rows, cols, name, tables):
    with open(tables / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    (tables / f"{name}.md").write_text("\n".join(lines) + "\n")


def fmt_time(s):
    if s is None:
        return "–"
    if s < 1e-3:
        return f"{s * 1e6:.1f} µs"
    if s < 1:
        return f"{s * 1e3:.2f} ms"
    return f"{s:.3f} s"


def style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def summarise(raw):
    """(experiment, method, n, p) -> stats dict from the measured repetitions."""
    groups = defaultdict(list)
    cpu = defaultdict(list)
    for r in raw:
        if r["kind"] != "measured" or r["status"] != "ok":
            continue
        k = (r["experiment"], r["method"], int(r["n"]), int(r["threads"]))
        groups[k].append(float(r["time_per_mult_s"]))
        cpu[k].append(float(r["cpu_utilisation"]))
    out = {}
    for k, ts in groups.items():
        q1, med, q3 = np.percentile(ts, [25, 50, 75])
        out[k] = {"median": med, "q1": q1, "q3": q3, "iqr": q3 - q1, "reps": len(ts),
                  "cpu_util": float(np.median(cpu[k]))}
    return out


def main():
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_run()
    run = run if run.is_absolute() else (ROOT / run)
    raw = load_csv(run / "raw_measurements.csv")
    runs = load_csv(run / "runs.csv")
    env = json.load(open(run / "environment.json"))
    tables, figs = ROOT / "report" / "tables", ROOT / "report" / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    stats = summarise(raw)
    runinfo = {(r["experiment"], r["method"], int(r["n"]), int(r["threads"])): r for r in runs}
    cores = env["cpu"]["physical_cores"]
    logical = env["cpu"]["logical_processors"]
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "font.family": "DejaVu Sans"})

    # ------------------------------------------------------------ environment
    c = env["cpu"]
    env_rows = [
        {"item": "CPU", "value": c["model"]},
        {"item": "Physical cores / logical processors", "value": f"{cores} / {logical} (Hyper-Threading)"},
        {"item": "Caches (L1d / L2 per core, L3 shared)",
         "value": f"{int(c.get('hw.l1dcachesize', 0)) // 1024} KiB / {int(c.get('hw.l2cachesize', 0)) // 1024} KiB / "
                  f"{int(c.get('hw.l3cachesize', 0)) // 2**20} MiB"},
        {"item": "SIMD", "value": ", ".join(c.get("simd_features", []))},
        {"item": "RAM", "value": f"{env['ram_bytes'] / 2**30:.0f} GiB"},
        {"item": "OS", "value": f"{env['os']} (Darwin {env['kernel']})"},
        {"item": "Java", "value": env["java_version"].splitlines()[0]},
        {"item": "JVM options", "value": " ".join(env["java_opts"])},
        {"item": "Power", "value": " ".join(c.get("power_source", []))},
        {"item": "Affinity", "value": env.get("affinity_note", "")},
    ]
    write_table(env_rows, ["item", "value"], "environment", tables)

    # --------------------------------------------------------- scaling table
    sizes = sorted({k[2] for k in stats if k[0] == "scaling"})
    threads = sorted({k[3] for k in stats if k[0] == "scaling" and k[1] != "seq"})
    rows = []
    for n in sizes:
        seq = stats.get(("scaling", "seq", n, 1))
        for m, ps in [("seq", [1])] + [(m, threads) for m in ("static", "stream")]:
            for p in ps:
                s = stats.get(("scaling", m, n, p))
                info = runinfo.get(("scaling", m, n, p), {})
                row = {"n": n, "method": m, "threads": p,
                       "oversubscribed": "yes" if p > logical else "",
                       "status": info.get("status", "missing")}
                if s:
                    row.update(median=fmt_time(s["median"]), median_s=f"{s['median']:.6g}",
                               iqr_pct=f"{100 * s['iqr'] / s['median']:.1f}", reps=s["reps"],
                               gflops=f"{2 * n ** 3 / s['median'] / 1e9:.2f}",
                               cpu_util=f"{s['cpu_util']:.2f}",
                               workers=info.get("distinct_worker_threads", ""),
                               peak_rss_mib=f"{int(info['peak_rss_bytes']) / 2**20:.0f}")
                    if seq:
                        S = seq["median"] / s["median"]
                        row.update(speedup=f"{S:.2f}", efficiency=f"{S / p:.2f}",
                                   speedup_lo=f"{seq['q1'] / s['q3']:.2f}",
                                   speedup_hi=f"{seq['q3'] / s['q1']:.2f}")
                rows.append(row)
    cols = ["n", "method", "threads", "oversubscribed", "status", "median", "iqr_pct", "reps",
            "speedup", "speedup_lo", "speedup_hi", "efficiency", "gflops", "cpu_util", "workers",
            "peak_rss_mib", "median_s"]
    write_table(rows, cols, "scaling_all", tables)

    # compact speedup matrix: one row per (method, n), one column per p
    mat = []
    for m in ("static", "stream"):
        for n in sizes:
            seq = stats.get(("scaling", "seq", n, 1))
            row = {"method": m, "n": n, "T_seq": fmt_time(seq["median"]) if seq else "–"}
            for p in threads:
                s = stats.get(("scaling", m, n, p))
                row[f"p={p}"] = f"{seq['median'] / s['median']:.2f}" if (s and seq) else "–"
            mat.append(row)
    write_table(mat, ["method", "n", "T_seq"] + [f"p={p}" for p in threads], "speedup_matrix", tables)

    # overhead of the pool with one worker
    ov = []
    for n in sizes:
        seq = stats.get(("scaling", "seq", n, 1))
        row = {"n": n, "T_seq": fmt_time(seq["median"]) if seq else "–"}
        for m in ("static", "stream"):
            s = stats.get(("scaling", m, n, 1))
            if s and seq:
                row[f"{m}_p1"] = fmt_time(s["median"])
                row[f"{m}_overhead_pct"] = f"{100 * (s['median'] / seq['median'] - 1):+.1f}"
        ov.append(row)
    write_table(ov, ["n", "T_seq", "static_p1", "static_overhead_pct", "stream_p1",
                     "stream_overhead_pct"], "one_thread_overhead", tables)

    # limits / non-ok configurations
    lim = [{"experiment": r["experiment"], "method": r["method"], "n": r["n"], "threads": r["threads"],
            "status": r["status"]} for r in runs if r["status"] != "ok"]
    largest = max((int(r["n"]) for r in runs if r["status"] == "ok"), default=None)
    lim.append({"experiment": "all", "method": "all", "n": largest, "threads": "all",
                "status": "largest n completed by every configuration"
                if not [r for r in runs if r["status"] != "ok"] else "largest n completed"})
    write_table(lim, ["experiment", "method", "n", "threads", "status"], "limits", tables)

    # memory
    mem = []
    for n in sizes:
        for m, ps in [("seq", [1])] + [(m, threads) for m in ("static", "stream")]:
            for p in ps:
                info = runinfo.get(("scaling", m, n, p))
                if not info or info["status"] != "ok":
                    continue
                mem.append({"n": n, "method": m, "threads": p,
                            "array_storage_mib": f"{int(info['array_storage_bytes']) / 2**20:.1f}",
                            "peak_rss_mib": f"{int(info['peak_rss_bytes']) / 2**20:.0f}",
                            "heap_used_mib": f"{int(info['java_heap_used_bytes']) / 2**20:.0f}",
                            "jvm_peak_threads": info["jvm_peak_thread_count"]})
    write_table(mem, ["n", "method", "threads", "array_storage_mib", "peak_rss_mib", "heap_used_mib",
                      "jvm_peak_threads"], "memory", tables)

    # ------------------------------------------------------------ vector table
    vsizes = sorted({k[2] for k in stats if k[0] == "vector"})
    vthreads = sorted({k[3] for k in stats if k[0] == "vector"})
    vrows = []
    for n in vsizes:
        for p in vthreads:
            base = "seq" if p == 1 else "static"
            nov = stats.get(("vector", base + "_novec", n, p))
            aut = stats.get(("vector", base, n, p))
            for var, m in (("novec", base + "_novec"), ("auto", base), ("vapi", base + "_vapi")):
                s = stats.get(("vector", m, n, p))
                info = runinfo.get(("vector", m, n, p), {})
                row = {"n": n, "threads": p, "variant": var, "method": m,
                       "status": info.get("status", "missing")}
                if s:
                    row.update(median=fmt_time(s["median"]), median_s=f"{s['median']:.6g}",
                               iqr_pct=f"{100 * s['iqr'] / s['median']:.1f}",
                               gflops=f"{2 * n ** 3 / s['median'] / 1e9:.2f}",
                               vs_novec=f"{nov['median'] / s['median']:.2f}" if nov else "",
                               vs_auto=f"{aut['median'] / s['median']:.2f}" if aut else "",
                               peak_rss_mib=f"{int(info['peak_rss_bytes']) / 2**20:.0f}",
                               lanes=info.get("vector_lanes", ""))
                vrows.append(row)
    write_table(vrows, ["n", "threads", "variant", "method", "status", "median", "iqr_pct", "gflops",
                        "vs_novec", "vs_auto", "lanes", "peak_rss_mib", "median_s"], "vector", tables)

    # ================================================================ figures
    ps = np.array(threads)

    def shade_hw(ax):
        ax.axvspan(logical + 0.5, max(threads) + 0.5, color=GRID, alpha=0.6, linewidth=0)
        ax.axvline(cores, color=INK2, linewidth=0.8, linestyle=":")
        ax.axvline(logical, color=INK2, linewidth=0.8, linestyle=":")
        top = ax.get_ylim()[1]
        ax.text(cores, top, f" {cores} cores", color=INK2, fontsize=7, va="top")
        ax.text(logical, top, f" {logical} logical", color=INK2, fontsize=7, va="top")
        ax.text(max(threads) + 0.4, top * 0.93, "oversubscribed",
                color=INK2, fontsize=7, ha="right", va="top")

    # Fig 1: speedup vs p, one panel per method, one line per n
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    for ax, m in zip(axes, ("static", "stream")):
        ax.plot(ps, ps, color=INK2, linewidth=1, linestyle="--", label="Ideal S = p")
        for i, n in enumerate(sizes):
            seq = stats.get(("scaling", "seq", n, 1))
            pts = [(p, seq["median"] / stats[("scaling", m, n, p)]["median"]) for p in threads
                   if seq and ("scaling", m, n, p) in stats]
            if not pts:
                continue
            x, y = zip(*pts)
            col = RAMP[min(i, len(RAMP) - 1)]
            ax.plot(x, y, color=col, marker="o", markersize=4, linewidth=2, label=f"n = {n}")
        ax.set_title(METHOD_LABEL[m])
        ax.set_xlabel("Threads p")
        ax.set_xticks(ps)
        ax.set_ylim(0, 3.2)     # the ideal line S = p continues above (clipped)
        style_axes(ax)
        shade_hw(ax)
    axes[0].set_ylabel("Speedup S(p) = T_seq / T_p  (medians)")
    axes[1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(figs / "speedup.png", dpi=200)
    plt.close(fig)

    # Fig 2: efficiency vs p
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    for ax, m in zip(axes, ("static", "stream")):
        ax.axhline(1, color=INK2, linewidth=1, linestyle="--")
        for i, n in enumerate(sizes):
            seq = stats.get(("scaling", "seq", n, 1))
            pts = [(p, seq["median"] / stats[("scaling", m, n, p)]["median"] / p) for p in threads
                   if seq and ("scaling", m, n, p) in stats]
            if not pts:
                continue
            x, y = zip(*pts)
            ax.plot(x, y, color=RAMP[min(i, len(RAMP) - 1)], marker="o", markersize=4, linewidth=2,
                    label=f"n = {n}")
        ax.set_title(METHOD_LABEL[m])
        ax.set_xlabel("Threads p")
        ax.set_xticks(ps)
        ax.set_ylim(0, 1.15)
        style_axes(ax)
        shade_hw(ax)
    axes[0].set_ylabel("Efficiency E(p) = S(p) / p")
    axes[1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(figs / "efficiency.png", dpi=200)
    plt.close(fig)

    # Fig 3: runtime vs n (log-log) for seq and both methods at p = logical
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for m, p in (("seq", 1), ("static", logical), ("stream", logical)):
        pts = [(n, stats[("scaling", m, n, p)]) for n in sizes if ("scaling", m, n, p) in stats]
        if not pts:
            continue
        x = np.array([a for a, _ in pts])
        med = np.array([s["median"] for _, s in pts])
        lo = med - np.array([s["q1"] for _, s in pts])
        hi = np.array([s["q3"] for _, s in pts]) - med
        col, mk, ls = METHOD_STYLE[m]
        lab = METHOD_LABEL[m] + ("" if m == "seq" else f", p = {p}")
        ax.errorbar(x, med, yerr=[lo, hi], color=col, marker=mk, markersize=5, linewidth=2,
                    linestyle=ls, capsize=2, label=lab)
    ref = stats.get(("scaling", "seq", sizes[-1], 1))
    if ref:
        xs = np.array(sizes, dtype=float)
        ax.plot(xs, ref["median"] * (xs / sizes[-1]) ** 3, color=GRID, linewidth=3, zorder=0,
                label="n³ reference")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.set_xlabel("Matrix size n")
    ax.set_ylabel("Time per multiplication (s, median; bars = IQR)")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / "runtime_vs_n.png", dpi=200)
    plt.close(fig)

    # Fig 4: CPU utilisation vs p (largest n), shows how many cores are really busy
    nbig = sizes[-1]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for m in ("static", "stream"):
        pts = [(p, stats[("scaling", m, nbig, p)]["cpu_util"]) for p in threads
               if ("scaling", m, nbig, p) in stats]
        if pts:
            x, y = zip(*pts)
            col, mk, ls = METHOD_STYLE[m]
            ax.plot(x, y, color=col, marker=mk, markersize=5, linewidth=2, linestyle=ls,
                    label=METHOD_LABEL[m])
    ax.plot(ps, np.minimum(ps, logical), color=INK2, linewidth=1, linestyle="--",
            label="min(p, logical processors)")
    ax.set_xlabel("Threads p")
    ax.set_xticks(ps)
    ax.set_ylabel(f"CPU time / wall time (n = {nbig})")
    ax.set_ylim(0, logical + 0.8)
    style_axes(ax)
    shade_hw(ax)
    ax.legend(frameon=False, fontsize=7, loc="lower right", bbox_to_anchor=(1.0, 0.08))
    fig.tight_layout()
    fig.savefig(figs / "cpu_utilisation.png", dpi=200)
    plt.close(fig)

    # Fig 5: vectorisation, grouped bars of GFLOP/s, one panel per thread count
    if vsizes:
        fig, axes = plt.subplots(1, len(vthreads), figsize=(4.6 * len(vthreads), 3.6), sharey=True)
        axes = np.atleast_1d(axes)
        w = 0.26
        for ax, p in zip(axes, vthreads):
            base = "seq" if p == 1 else "static"
            x = np.arange(len(vsizes))
            for j, (var, suf) in enumerate((("novec", "_novec"), ("auto", ""), ("vapi", "_vapi"))):
                vals = [2 * n ** 3 / stats[("vector", base + suf, n, p)]["median"] / 1e9
                        if ("vector", base + suf, n, p) in stats else 0 for n in vsizes]
                col, lab = VARIANT_STYLE[var]
                bars = ax.bar(x + (j - 1) * w, vals, w - 0.03, color=col, label=lab,
                              edgecolor="white", linewidth=1)
                for b, v in zip(bars, vals):
                    if v:
                        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center",
                                va="bottom", fontsize=7, color=INK)
            ax.set_xticks(x)
            ax.set_xticklabels([f"n = {n}" for n in vsizes])
            ax.set_title(f"p = {p} ({'sequential kernel' if p == 1 else 'static parallel kernel'})")
            style_axes(ax)
            ax.grid(axis="x", visible=False)
        axes[0].set_ylabel("GFLOP/s (2n³ / median time)")
        axes[-1].legend(frameon=False, fontsize=7, loc="upper right")
        fig.tight_layout()
        fig.savefig(figs / "vectorisation.png", dpi=200)
        plt.close(fig)

    # Fig 6: memory, peak RSS vs p at the largest n, with the array storage line
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for m in ("static", "stream"):
        pts = [(p, int(runinfo[("scaling", m, nbig, p)]["peak_rss_bytes"]) / 2**20) for p in threads
               if runinfo.get(("scaling", m, nbig, p), {}).get("status") == "ok"]
        if pts:
            x, y = zip(*pts)
            col, mk, ls = METHOD_STYLE[m]
            ax.plot(x, y, color=col, marker=mk, markersize=5, linewidth=2, linestyle=ls,
                    label=METHOD_LABEL[m])
    sq = runinfo.get(("scaling", "seq", nbig, 1))
    if sq and sq["status"] == "ok":
        ax.axhline(int(sq["peak_rss_bytes"]) / 2**20, color=INK2, linestyle=":", linewidth=1.2,
                   label="Sequential (peak RSS)")
        ax.axhline(int(sq["array_storage_bytes"]) / 2**20, color=GRID, linewidth=3,
                   label="Array storage A + B + C")
    ax.set_xlabel("Threads p")
    ax.set_xticks(ps)
    ax.set_ylabel(f"Peak RSS of the JVM (MiB, n = {nbig})")
    ax.set_ylim(0, None)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(figs / "memory.png", dpi=200)
    plt.close(fig)

    print(f"Analysed {run.relative_to(ROOT)}: {len(stats)} configurations")
    print(f"Tables -> {tables.relative_to(ROOT)}/, figures -> {figs.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
