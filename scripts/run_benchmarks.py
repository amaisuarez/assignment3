#!/usr/bin/env python3
"""Benchmark harness for Assignment 3 (adapted from the Assignment 2 harness).

Two experiments, both in Java, float64, i-k-j kernel, same inputs as A1/A2:

  scaling  thread-count study. For every size n: the sequential baseline `seq`
           and every parallel method (`static`, `stream`) with every thread count
           in the config (including p = 1 and counts above the 4 logical
           processors, i.e. oversubscription).
  vector   optional SIMD extension. At fixed thread counts (p = 1 -> sequential
           kernel, p > 1 -> static parallel kernel) three code-generation variants
           of the SAME kernel:
             novec  JIT auto-vectorisation disabled (-XX:-UseSuperWord)
             auto   default JIT (auto-vectorisation allowed), = seq / static
             vapi   explicit SIMD with the Java Vector API (java/VecKernels.java)
           so the effect of vectorisation is measured with the thread count fixed.

Every configuration runs in a fresh JVM that loads the inputs, creates the thread
pool, does W warm-up rounds (each >= batch_target_ms), then R measured repetitions
of `batch` multiplications. Timing boundary = one complete multiplication call,
including task hand-off to the pool and waiting for all tasks, excluding input
loading, allocation of C and pool creation (see java/MatMul3.java). The order of
the configurations is shuffled within each size with a fixed seed, so slow drifts
(temperature, background activity) do not always hit the same method.

Budgets (declared before running, in the config):
  * a process is killed if one output line (one warm-up round or one measured
    repetition) takes longer than per_repetition_budget_s (+ a start-up allowance
    for the first line) -> status "timeout";
  * if the estimated array storage exceeds memory_budget_bytes the run is not
    started -> "skipped_memory_budget".
  After a failure, larger sizes are not attempted for that (method, threads)
  -> "not_attempted_after_<status>". No time is ever invented for them.

Memory: peak resident set size of the JVM process from os.wait4() (ru_maxrss), the
analytical array storage of A, B and C (3 * 8 n^2 bytes) and the Java heap in use
at the end. CPU: process CPU time during every timed repetition (from the JVM) and
user + system time of the whole process (from wait4).

Output: results/run_<timestamp>/{raw_measurements.csv, runs.csv, environment.json,
        config.json, stderr/}
"""
import argparse
import csv
import json
import os
import random
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import collect_env  # noqa: E402
import gen_matrices  # noqa: E402

RAW_FIELDS = ["run_id", "experiment", "method", "variant", "n", "threads", "dtype", "kind",
              "rep_index", "elapsed_ns", "batch", "time_per_mult_s", "process_cpu_ns",
              "cpu_utilisation", "checksum", "status"]
RUN_FIELDS = ["run_id", "experiment", "method", "variant", "n", "threads", "oversubscribed",
              "status", "exit_code", "warmups", "repetitions", "batch", "tasks_per_multiplication",
              "distinct_worker_threads", "jvm_peak_thread_count", "available_processors",
              "vector_lanes", "array_storage_bytes", "peak_rss_bytes", "memory_metric",
              "java_heap_used_bytes", "estimated_bytes", "process_user_s", "process_sys_s",
              "wall_time_s", "line_timeout_s", "order_in_group", "command"]

# method name in the study -> (method passed to MatMul3, extra JVM flags)
METHODS = {
    "seq": ("seq", []),
    "seq_novec": ("seq", ["-XX:-UseSuperWord"]),
    "seq_vapi": ("seq_vapi", []),
    "static": ("static", []),
    "static_novec": ("static", ["-XX:-UseSuperWord"]),
    "static_vapi": ("static_vapi", []),
    "stream": ("stream", []),
}
VARIANT_SUFFIX = {"novec": "_novec", "auto": "", "vapi": "_vapi"}


def warmups_for(cfg, n):
    for rule in cfg["warmup_rules"]:
        if n <= rule["max_n"]:
            return rule["warmups"]
    return cfg["warmup_rules"][-1]["warmups"]


def estimate_bytes(n):
    """Planning estimate of array storage (bytes): A, B and C as dense float64."""
    return 3 * 8 * n * n


def command(cfg, method, n, p, pa, pb, warmup, reps):
    kernel, extra = METHODS[method]
    return (["java"] + cfg["java_opts"] + extra + ["-cp", str(ROOT / "java" / "build"), "MatMul3",
            "--method", kernel, "--n", str(n), "--threads", str(p), "--a", str(pa), "--b", str(pb),
            "--warmup", str(warmup), "--reps", str(reps), "--batch", "0",
            "--target-ms", str(cfg["batch_target_ms"])])


def run_process(cmd, line_budget_s, startup_s, stderr_path):
    """Run cmd; kill it if one output line takes longer than the budget.

    Returns (stdout_lines, exit_code, timed_out, rusage, wall_s)."""
    lines = []
    last = [time.monotonic()]
    with open(stderr_path, "w") as err:
        t0 = time.monotonic()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, text=True)

        def reader():
            for line in p.stdout:
                lines.append(line.rstrip("\n"))
                last[0] = time.monotonic()

        th = threading.Thread(target=reader, daemon=True)
        th.start()
        timed_out = False
        while th.is_alive():
            th.join(0.2)
            allowed = line_budget_s + (startup_s if not lines else 0)
            if th.is_alive() and time.monotonic() - last[0] > allowed:
                timed_out = True
                p.kill()
                th.join()
        _, status, usage = os.wait4(p.pid, 0)
        wall = time.monotonic() - t0
        p.returncode = os.waitstatus_to_exitcode(status)
        p.stdout.close()
    return lines, p.returncode, timed_out, usage, wall


def jobs(cfg):
    """Ordered list of groups (one per experiment and size), shuffled inside."""
    rng = random.Random(cfg["seed"])
    groups = []
    s = cfg.get("scaling")
    if s:
        for n in s["sizes"]:
            g = [dict(experiment="scaling", method="seq", variant="auto", n=n, threads=1)]
            g += [dict(experiment="scaling", method=m, variant="auto", n=n, threads=p)
                  for m in s["methods"] for p in s["threads"]]
            rng.shuffle(g)
            groups.append(g)
    v = cfg.get("vector")
    if v:
        for n in v["sizes"]:
            g = []
            for p in v["threads"]:
                base = "seq" if p == 1 else "static"
                g += [dict(experiment="vector", method=base + VARIANT_SUFFIX[var], variant=var, n=n,
                           threads=p) for var in v["variants"]]
            rng.shuffle(g)
            groups.append(g)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.json"))
    ap.add_argument("--only", nargs="*", help="run only these experiments (scaling, vector)")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    if args.only:
        for key in ("scaling", "vector"):
            if key not in args.only:
                cfg.pop(key, None)

    if not (ROOT / "java" / "build" / "MatMul3.class").exists():
        sys.exit("Build first: ./build.sh")

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = ROOT / "results" / f"run_{run_id}"
    (out_dir / "stderr").mkdir(parents=True)
    json.dump(cfg, open(out_dir / "config.json", "w"), indent=2)
    env = collect_env.collect(cfg)
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2))
    logical = env["cpu"]["logical_processors"]

    raw_f = open(out_dir / "raw_measurements.csv", "w", newline="")
    runs_f = open(out_dir / "runs.csv", "w", newline="")
    raw = csv.DictWriter(raw_f, RAW_FIELDS, lineterminator="\n")
    runs = csv.DictWriter(runs_f, RUN_FIELDS, lineterminator="\n")
    raw.writeheader()
    runs.writeheader()

    R = cfg["repetitions"]
    failed = {}            # (experiment, method, threads) -> (n, status)
    t_start = time.monotonic()

    for group in jobs(cfg):
        for pos, job in enumerate(group):
            exp, m, var, n, p = job["experiment"], job["method"], job["variant"], job["n"], job["threads"]
            pa, pb = gen_matrices.ensure(n, cfg["seed"])
            W = warmups_for(cfg, n)
            est = estimate_bytes(n)
            base = {"run_id": run_id, "experiment": exp, "method": m, "variant": var, "n": n,
                    "threads": p, "oversubscribed": p > logical, "warmups": W, "repetitions": R,
                    "estimated_bytes": est, "line_timeout_s": cfg["per_repetition_budget_s"],
                    "order_in_group": pos}
            label = f"[{exp:7s} n={n:5d} p={p}] {m:13s}"

            key = (exp, m, p)
            if key in failed and n >= failed[key][0]:
                runs.writerow({**base, "status": f"not_attempted_after_{failed[key][1]}"})
                print(f"{label} not attempted (earlier {failed[key][1]})")
                continue
            if est > cfg["memory_budget_bytes"]:
                runs.writerow({**base, "status": "skipped_memory_budget"})
                failed[key] = (n, "memory_budget")
                print(f"{label} skipped (estimated {est / 1e9:.2f} GB > budget)")
                continue

            cmd = command(cfg, m, n, p, pa, pb, W, R)
            print(f"{label} W={W} R={R} ...", end=" ", flush=True)
            lines, code, timed_out, usage, wall = run_process(
                cmd, cfg["per_repetition_budget_s"], cfg["process_startup_allowance_s"],
                out_dir / "stderr" / f"{exp}_{m}_n{n}_p{p}.txt")
            status = "timeout" if timed_out else ("ok" if code == 0 else "error")

            scale = 1 if sys.platform == "darwin" else 1024   # Linux reports KiB, macOS bytes
            meta = {"peak_rss_bytes": usage.ru_maxrss * scale, "memory_metric": "ru_maxrss_wait4",
                    "process_user_s": round(usage.ru_utime, 3), "process_sys_s": round(usage.ru_stime, 3)}
            batch = ""
            for line in lines:
                if line.startswith("#"):
                    k, value = line[1:].split(",")
                    meta[k] = int(value)
                    continue
                kind, idx, ns, b, chk, cpu = line.split(",")
                if kind == "measured":
                    batch = int(b)
                raw.writerow({"run_id": run_id, "experiment": exp, "method": m, "variant": var,
                              "n": n, "threads": p, "dtype": "float64", "kind": kind,
                              "rep_index": int(idx), "elapsed_ns": int(ns), "batch": int(b),
                              "time_per_mult_s": int(ns) / 1e9 / int(b), "process_cpu_ns": int(cpu),
                              "cpu_utilisation": round(int(cpu) / int(ns), 3), "checksum": chk,
                              "status": status})
            meta.pop("threads_requested", None)
            runs.writerow({**base, **meta, "status": status, "exit_code": code, "batch": batch,
                           "wall_time_s": round(wall, 3), "command": " ".join(cmd)})
            raw_f.flush()
            runs_f.flush()
            print(f"{status} ({wall:.1f}s, peak RSS {meta['peak_rss_bytes'] / 2**20:.0f} MiB, "
                  f"{meta.get('distinct_worker_threads', '?')} worker threads)")
            if status != "ok":
                failed[key] = (n, status)

    raw_f.close()
    runs_f.close()
    mins = (time.monotonic() - t_start) / 60
    print(f"\nFinished in {mins:.1f} min. Results written to {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
