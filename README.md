# Parallel matrix multiplication in Java

**Big Data – Individual Assignment 3** · Amai Suárez Navarro · Grado en Ciencia e Ingeniería de Datos (ULPGC), 2026-2027

This folder has everything behind my third Big Data assignment. It continues
Assignment 2 (https://github.com/amaisuarez/assignment2) and asks:

> How does the number of threads change the runtime of a shared-memory matrix
> multiplication in Java on my laptop, and when does running in parallel help?

Optional extension: how much does SIMD vectorisation add, measured with the thread
count fixed?

The full write-up is in `Assignment03_SuarezNavarro_Report.pdf`.

**Repository:** https://github.com/amaisuarez/assignment3

---

## What I found (short version)

All numbers come from my laptop, an Intel Core i5-5350U MacBook Air with **2 physical
cores, 4 logical processors** and 8 GB of RAM (run `results/run_20261104-161358/`).

- **Two threads roughly halve the time.** For n ≥ 256 the speedup with p = 2 was
  1.5–2.0 (efficiency 0.77–0.99). At n = 2048: 6.94 s sequential → 3.95 s.
- **Hyper-Threading barely helps here.** p = 4 keeps all 4 logical processors busy
  (CPU time / wall time = 3.9) but is only 3 % faster than p = 2 at n = 2048 (at most
  16 % at other sizes). The two threads on one core compete for the same units.
- **More threads than the hardware never helps.** p = 6 and 8 were the same or worse
  (−16 % at n = 2048, p = 8).
- **Small matrices don't benefit.** At n = 64 the pool alone adds 20 % with one
  thread, and the best speedup was 1.2.
- **Static blocks and ForkJoin streams perform the same** for large n; the stream has
  more overhead for medium sizes with one thread.
- **SIMD (optional extension):** the JIT's auto-vectorisation made no difference, but
  explicit SIMD with the Vector API was 1.4–2.2× faster at the same thread count.
  Threads + SIMD: n = 1024 went from 965 ms to 242 ms (3.98×).
- Speedups > 2 at n = 1536 turned out **not** to be superlinear: re-runs showed
  interference in the sequential runs and a reproducible ~10 % JIT difference at that
  size (`config_recheck_1536.json`, report Section 4.3).

All methods give **bitwise-identical results** to the sequential kernel.

---

## What I reuse from Assignment 2

| Reused | From [`amaisuarez/assignment2`](https://github.com/amaisuarez/assignment2) |
|---|---|
| Sequential baseline | `java/Dense.java`, method `ikj` (loop order i-k-j), copied unchanged into `java/Kernels.java` (`Kernels.ikj`). It was the fastest sequential method in A2 |
| Inputs | `scripts/gen_matrices.py` and `inputs/manifest.csv`, unchanged (originally from Assignment 1). Same seed `20262027`, so the files are byte-identical to the A1/A2 inputs (the script checks the SHA-256) |
| Harness | `scripts/run_benchmarks.py`, `scripts/analyze.py`, `tests/validate.py`, `java/MatMul2.java` → `MatMul3.java`: same protocol (fresh JVM per configuration, warm-ups, batching, 5 repetitions, budgets), sparse parts removed, threads added |
| Environment info | `scripts/collect_env.py`, with cache sizes, SIMD flags, power source and an affinity note added |
| JVM settings | `-XX:+UseSerialGC -Xms256m -Xmx4g` (plus `--add-modules jdk.incubator.vector` for the SIMD extension) |

---

## The methods

All methods compute C = A·B for n × n float64 matrices stored as one row-major
`double[]`, with the same i-k-j loop order and the same inputs.

| Method | What it is | File |
|---|---|---|
| `seq` | **Sequential baseline**: A2 i-k-j kernel, unchanged | `java/Kernels.java` |
| `static` | **Parallel, static partition**: fixed thread pool (`ExecutorService`) with *p* threads; the n rows of C are cut into *p* contiguous blocks (sizes differ by ≤ 1 row), one task per block | `java/Parallel.java` |
| `stream` | **Parallel, dynamic scheduling**: `ForkJoinPool` with parallelism *p* running `IntStream.range(0, n).parallel()` over the rows; the framework splits the range and balances it by work stealing | `java/Parallel.java` |
| `seq_novec`, `static_novec` | Same kernels with the JIT's auto-vectorisation off (`-XX:-UseSuperWord`) | — (JVM flag) |
| `seq_vapi`, `static_vapi` | Same kernels with the inner loop written in explicit SIMD with the Java **Vector API** (4 doubles per instruction with AVX2) | `java/VecKernels.java` |

Every thread writes only its own rows of C and only reads A and B, so there are no
data races and no locks. Every C[i][j] is still the sum over k = 0…n−1 in the same
order, so **all methods give bit-identical results** (checked by `tests/validate.py`).

---

## What's in the folder

```
java/Kernels.java          sequential i-k-j (A2) and the row-range kernel used by the threads
java/Parallel.java         parallel versions: static row blocks, parallel stream (ForkJoin)
java/VecKernels.java       explicit SIMD row kernel (Vector API, optional extension)
java/MatMul3.java          benchmark driver (one process = one configuration)

scripts/gen_matrices.py    dense inputs (reused from A1/A2, same files)
scripts/run_benchmarks.py  runs the whole study
scripts/collect_env.py     saves info about the machine (CPU, caches, RAM, versions...)
scripts/analyze.py         builds the tables and plots from the raw data
tests/validate.py          correctness checks (run before benchmarking)

config.json                settings of the real study (sizes, thread counts, repetitions, limits)
config_quick.json          1-minute smoke test, just to check that everything works
config_recheck_1536.json   extra re-runs of n = 1536 (report Section 4.3)
build.sh                   compiles the Java code
build_report.sh            builds the PDF from report/report.md (pandoc + headless Chrome)

inputs/manifest.csv        seeds and SHA-256 of the inputs (same as A1/A2)
results/                   raw data of the study, its log and the validation report
report/                    report source (report.md), tables and figures generated by analyze.py
Assignment03_SuarezNavarro_Report.pdf   the report
```

---

## What you need

- macOS or Linux (on Windows, use WSL)
- **Java JDK 21** (I used OpenJDK 21.0.6). The SIMD extension uses the Vector API,
  an incubator module in JDK 21; `build.sh` and the configs add it with
  `--add-modules jdk.incubator.vector` (the JVM prints a warning about it, that's normal)
- Python 3.9 or newer, with NumPy (only to check results) and Matplotlib (plots):

```bash
pip install -r requirements.txt
```

---

## How to run it

**1. Compile**

```bash
bash build.sh
```

**2. Check correctness** (about 1–2 minutes)

```bash
python3 tests/validate.py | tee results/validation_report.txt
```

It must finish with `ALL TESTS PASSED`.

**3. (Optional) Quick test, about 1 minute**

```bash
python3 scripts/run_benchmarks.py --config config_quick.json
```

**4. The real study** (about 20 minutes on my laptop)

Keep the laptop plugged in and don't use it meanwhile; other programs steal cores
and change the parallel results. On a Mac, `caffeinate` stops it from sleeping:

```bash
caffeinate -i python3 scripts/run_benchmarks.py    # on Linux, drop "caffeinate -i"
```

You can run the two experiments separately with `--only scaling` or `--only vector`.
At the end it prints the folder with the results, like `results/run_YYYYMMDD-HHMMSS`.

**5. Tables and plots**

```bash
python3 scripts/analyze.py results/run_YYYYMMDD-HHMMSS
```

This writes the tables to `report/tables/` and the plots to `report/figures/`.

The two n = 1536 re-runs of the report (Section 4.3) were made with
`python3 scripts/run_benchmarks.py --config config_recheck_1536.json`, run twice.

**6. (Optional) The PDF**

```bash
bash build_report.sh      # needs pandoc and Chrome/Chromium
```

The input matrices are generated automatically in `data/` (ignored by git because
they are large) and always come out the same thanks to the fixed seed.

---

## How I made the comparisons fair

| | What I did |
|---|---|
| Inputs | The A1/A2 dense files: values uniform in [−1, 1), seed `20262027`, float64. Every method and thread count multiplies exactly the same matrices |
| Baseline | `seq` is the same algorithm, loop order, storage and numeric type as the parallel kernels. The parallel kernels with *p* = 1 are also measured, to see the cost of the thread pool itself |
| Sizes | n = 64, 128, 256, 512, 1024, 1536, 2048 |
| Threads | p = 1, 2, 3, 4, 6, 8. The laptop has 2 physical cores and 4 logical processors (Hyper-Threading), so p = 6 and 8 are **oversubscribed** |
| Thread control | *p* = number of worker threads of the pool created by my code (the main thread only waits). No numerical library, no common ForkJoinPool. The JVM's own threads (JIT, 1 SerialGC thread) are counted and recorded (`jvm_peak_thread_count`). The driver also records how many distinct workers really ran tasks |
| Affinity | None: macOS doesn't let programs pin threads to cores |
| What I time | One full multiplication call: for the parallel methods it includes handing the tasks to the pool and waiting for all of them. Reading the files, allocating C and creating the pool are outside the timer. Zeroing C is inside (part of the i-k-j kernel) |
| Timer | `System.nanoTime()` (monotonic); C's checksum is printed after every repetition so the JIT can't skip the work. Process CPU time is recorded for every repetition too |
| Warm-up | Fresh JVM per configuration; 5 warm-up rounds for n ≤ 512, 3 for n = 1024, 2 above, each lasting ≥ 100 ms. Not included in the statistics |
| Small matrices | Each repetition repeats the multiplication until it lasts ~100 ms and reports the time per multiplication |
| Repetitions | 5 measured repetitions per configuration; median and interquartile range (IQR). Speedup and efficiency use the medians |
| Order | Configurations shuffled within each size (fixed seed), to avoid drift bias |
| Memory | Peak RSS of the JVM process (`ru_maxrss` from `wait4`), analytical array storage of A, B, C (3·8n² bytes) and Java heap in use |
| Limits (set beforehand) | 60 s per repetition and 3 GB of array storage. A configuration that exceeds them is recorded as `timeout` / `skipped_memory_budget`, and larger sizes are not attempted for it |

---

## Raw data format

- `results/run_*/raw_measurements.csv`: one row per warm-up or measured repetition, with
  experiment, method, variant, n, threads, kind, time in ns, batch, time per
  multiplication, process CPU time, CPU utilisation and checksum.
- `results/run_*/runs.csv`: one row per configuration, with status, oversubscription,
  tasks, worker threads seen, JVM thread count, peak RSS, heap, user/system CPU time,
  budgets and the exact command.
- `results/run_*/environment.json`: machine and version details.

---

## Limitations

Everything was measured on one old dual-core laptop, so absolute times and the
exact shape of the speedup curves are specific to it. With only 2 physical cores, the
study can't show how the method scales on many cores. macOS has no thread pinning,
so the OS decides where threads run. Peak RSS mostly reflects how much heap the JVM
reserved, not how much data is alive.

---

## Use of AI

I used Claude (Anthropic) to help design the experiment, explain the techniques, generate
the first versions of the code and scripts, and draft the report from my results. I ran all
the experiments on my own computer, and every number in the report comes from those runs.
