#!/usr/bin/env python3
"""Correctness checks for Assignment 3, run BEFORE any benchmark.

Adapted from tests/validate.py of Assignment 2 (sparse checks removed, threads added).

1. Known answers: identity, zero and a hand-computed 2x2, for every method and
   several thread counts.
2. Sequential baseline vs a reference library: the seq kernel is compared with
   NumPy's A @ B (reference only, not one of the studied methods) using the
   forward error bound of a dot product of length n (Higham, Accuracy and
   Stability of Numerical Algorithms, 2nd ed., Sec. 3.5):
       |C - C_exact| <= gamma_n (|A| |B|),   gamma_n = n u / (1 - n u),  u = 2^-53
   Both results satisfy it, so we accept |C - C_numpy| <= 2 gamma_n (|A| |B|).
3. Parallel == sequential, BIT FOR BIT, for every parallel method and thread
   count, including n smaller than p (some threads get no rows), n not divisible
   by p, and more threads than logical processors (oversubscription). The parallel
   kernels add the same products in the same k order for every C[i][j] and each
   row is written by one thread only, so any difference would reveal a data race
   or a partition error (a missing or duplicated row).
4. Repeatability: the same parallel configuration run several times in one
   process returns the same checksum every time (a race would make it vary).
5. Real thread count: the static partition must create exactly min(p, n)
   tasks per multiplication (no empty or missing blocks), and the number of
   distinct worker threads that executed tasks must be between 1 and p. (It can
   be below p for small n: a worker that finishes early may take the next queued
   task. The benchmark reports this number for every configuration.)
6. Inputs: dense files are byte-identical to the Assignment 1/2 inputs (SHA-256
   in inputs/manifest.csv).

Usage: python3 tests/validate.py      (after ./build.sh)
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import gen_matrices  # noqa: E402

TMP = ROOT / "tests" / "tmp"
CP = str(ROOT / "java" / "build")
JAVA_OPTS = json.load(open(ROOT / "config.json"))["java_opts"]
PAR_METHODS = ["static", "stream", "static_vapi"]
U = 2.0 ** -53
failures = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def write_dense(path, M):
    np.ascontiguousarray(M, dtype="<f8").tofile(path)


def run(method, n, pa, pb, p=1, reps=1):
    out = TMP / f"C_{method}_{n}_{p}.bin"
    cmd = (["java"] + JAVA_OPTS + ["-cp", CP, "MatMul3", "--method", method, "--n", str(n),
           "--threads", str(p), "--a", str(pa), "--b", str(pb), "--warmup", "0",
           "--reps", str(reps), "--batch", "1", "--out", str(out)])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{r.stderr}")
    meta, checksums = {}, []
    for line in r.stdout.splitlines():
        if line.startswith("#"):
            k, v = line[1:].split(",")
            meta[k] = int(v)
        elif line.startswith("measured"):
            checksums.append(line.split(",")[4])
    meta["checksums"] = checksums
    return np.fromfile(out, dtype="<f8").reshape(n, n), meta


def known_cases():
    print("1. Known answers")
    cases = {
        "identity x M": (np.eye(4), np.arange(16, dtype=float).reshape(4, 4) - 7.5),
        "M x identity": (np.arange(16, dtype=float).reshape(4, 4) * 0.25, np.eye(4)),
        "zero x M": (np.zeros((3, 3)), np.arange(9, dtype=float).reshape(3, 3)),
        "2x2 by hand": (np.array([[1., 2.], [3., 4.]]), np.array([[5., 6.], [7., 8.]])),
        "identity 9x9 (n > lanes)": (np.eye(9), np.arange(81, dtype=float).reshape(9, 9) / 3),
    }
    expected = {"2x2 by hand": np.array([[19., 22.], [43., 50.]])}
    for name, (A, B) in cases.items():
        n = A.shape[0]
        want = expected.get(name, A @ B)       # identity/zero cases are exact
        pa, pb = TMP / "ka.bin", TMP / "kb.bin"
        write_dense(pa, A)
        write_dense(pb, B)
        for m in ["seq", "seq_vapi"]:
            C, _ = run(m, n, pa, pb)
            check(np.array_equal(C, want), f"{name}: {m}")
        for m in PAR_METHODS:
            for p in [1, 2, 4]:
                C, _ = run(m, n, pa, pb, p)
                check(np.array_equal(C, want), f"{name}: {m} p={p}")


def random_inputs():
    print("2-3. Random inputs: seq vs NumPy; every parallel run == seq bit for bit")
    rng = np.random.default_rng(20262027)
    for n in [1, 2, 3, 5, 7, 16, 37, 64, 100, 131]:
        A = rng.uniform(-1, 1, (n, n))
        B = rng.uniform(-1, 1, (n, n))
        pa, pb = TMP / "ra.bin", TMP / "rb.bin"
        write_dense(pa, A)
        write_dense(pb, B)
        seq, _ = run("seq", n, pa, pb)
        gamma = n * U / (1 - n * U)
        bound = 2 * gamma * (np.abs(A) @ np.abs(B))
        err = np.abs(seq - A @ B)
        check(np.all(err <= bound), f"n={n:3d} seq vs NumPy: max err {err.max():.2e}, "
                                    f"max allowed {bound.max():.2e}")
        C, _ = run("seq_vapi", n, pa, pb)
        check(np.array_equal(C, seq), f"n={n:3d} seq_vapi == seq (bitwise)")
        for m in PAR_METHODS:
            bad = []
            for p in [1, 2, 3, 4, 6, 8]:
                C, meta = run(m, n, pa, pb, p)
                if not np.array_equal(C, seq):
                    bad.append(f"p={p} differs")
                used = meta["distinct_worker_threads"]
                if not 1 <= used <= p:
                    bad.append(f"p={p} used {used} threads")
                if m.startswith("static") and meta["tasks_per_multiplication"] != min(p, n):
                    bad.append(f"p={p} made {meta['tasks_per_multiplication']} tasks")
            check(not bad, f"n={n:3d} {m:11s} p=1,2,3,4,6,8 == seq (bitwise); 1 <= threads <= p"
                           + ("; tasks = min(p, n)" if m.startswith("static") else "")
                           + (f"  [{'; '.join(bad)}]" if bad else ""))


def repeatability():
    print("4. Repeatability of parallel runs (20 multiplications per process)")
    pa, pb = gen_matrices.ensure(128, 20262027)
    _, ref = run("seq", 128, pa, pb)
    for m in PAR_METHODS:
        for p in [4, 8]:
            _, meta = run(m, 128, pa, pb, p, reps=20)
            same = set(meta["checksums"]) == set(ref["checksums"])
            check(same, f"n=128 {m:11s} p={p}: 20 checksums all equal to seq "
                        f"({meta['checksums'][0]})")


def generators():
    print("6. Input generators")
    for n in [16, 64]:
        try:
            gen_matrices.ensure(n, 20262027)   # exits on SHA-256 mismatch with the manifest
            check(True, f"dense n={n}: byte-identical to the Assignment 1/2 inputs (SHA-256)")
        except SystemExit as e:
            check(False, f"dense n={n}: {e}")


def main():
    if not (ROOT / "java" / "build" / "MatMul3.class").exists():
        sys.exit("Build first: ./build.sh")
    TMP.mkdir(parents=True, exist_ok=True)
    known_cases()
    random_inputs()
    repeatability()
    generators()
    print()
    if failures:
        print(f"{len(failures)} TEST(S) FAILED")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
