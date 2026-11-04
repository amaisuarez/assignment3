import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Benchmark driver for Assignment 3: one process = one (method, n, threads) configuration.
 * Adapted from java/MatMul2.java of Assignment 2 (sparse parts removed, threads added).
 *
 * Methods (all i-k-j, float64, bit-identical results; see Kernels.java):
 *   seq          sequential baseline (Assignment 2 ikj, unchanged)
 *   seq_vapi     sequential, explicit SIMD with the Vector API      (optional extension)
 *   static       parallel, fixed thread pool, static row blocks    (Parallel.StaticRows)
 *   static_vapi  same, with the Vector API row kernel              (optional extension)
 *   stream       parallel, ForkJoinPool + parallel stream over rows (Parallel.StreamRows)
 *
 * Input: raw little-endian float64, row-major, n*n values, no header (same files as A1/A2).
 *
 * Sequence inside the process (only step 4 produces the reported samples):
 *   1. load A and B, allocate C, create the thread pool              (not timed)
 *   2. W warm-up rounds, each repeating the multiplication for >= target ms
 *   3. batch = max(1, ceil(target / t)) with t from the last warm-up round
 *      (or from one extra timed multiplication if W = 0)
 *   4. R measured repetitions, each timing `batch` multiplications
 *   5. one untimed multiplication that records which threads did work
 *
 * Timer: System.nanoTime() (monotonic). The timed region is the whole call that
 * performs the multiplication: for the parallel methods it includes handing the
 * tasks to the pool and waiting for all of them, but not creating the pool.
 * Zeroing C is inside the timer (it is part of the i-k-j kernel), allocating C is not.
 *
 * Sample line: kind,index,elapsed_ns,batch,checksum,process_cpu_ns
 *   checksum        sum of all elements of C, printed after every repetition so the
 *                   JIT cannot remove the computation
 *   process_cpu_ns  CPU time (user + system, all threads of the JVM) consumed during
 *                   the timed region, from com.sun.management.OperatingSystemMXBean
 * Lines starting with '#' carry metadata.
 */
public final class MatMul3 {

    static double[] loadDense(String path, int n) throws IOException {
        ByteBuffer buf = ByteBuffer.wrap(Files.readAllBytes(Path.of(path))).order(ByteOrder.LITTLE_ENDIAN);
        if (buf.capacity() != 8L * n * n)
            throw new IOException(path + ": expected " + (long) n * n + " float64 values");
        double[] m = new double[n * n];
        buf.asDoubleBuffer().get(m);
        return m;
    }

    static void saveDense(String path, double[] C) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(8 * C.length).order(ByteOrder.LITTLE_ENDIAN);
        buf.asDoubleBuffer().put(C);
        Files.write(Path.of(path), buf.array());
    }

    static double sum(double[] x) {
        double s = 0.0;
        for (double v : x) s += v;
        return s;
    }

    static final com.sun.management.OperatingSystemMXBean OS =
            (com.sun.management.OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean();

    static long cpuNs() { return OS.getProcessCpuTime(); }

    /** One configured method; run(k) performs k multiplications. */
    interface Kernel extends AutoCloseable {
        void run(int times) throws Exception;
        default void track(boolean on) {}
        default int distinctThreads() { return 1; }
        default int tasks() { return 1; }
        default void close() {}
    }

    static Kernel make(String method, double[] A, double[] B, double[] C, int n, int p) {
        return switch (method) {
            case "seq" -> times -> { for (int t = 0; t < times; t++) Kernels.ikj(A, B, C, n); };
            case "seq_vapi" -> times -> { for (int t = 0; t < times; t++) VecKernels.ikjRows(A, B, C, n, 0, n); };
            case "static", "static_vapi", "stream" -> {
                Parallel.RowKernel rk = Parallel.rowKernel(method.endsWith("_vapi"));
                Parallel.Runner r = method.equals("stream")
                        ? new Parallel.StreamRows(A, B, C, n, p, rk)
                        : new Parallel.StaticRows(A, B, C, n, p, rk);
                yield new Kernel() {
                    public void run(int times) throws Exception { for (int t = 0; t < times; t++) r.multiply(); }
                    public void track(boolean on) { r.track(on); }
                    public int distinctThreads() { return r.distinctThreads(); }
                    public int tasks() { return r.tasks(); }
                    public void close() { r.close(); }
                };
            }
            default -> throw new IllegalArgumentException("unknown method " + method);
        };
    }

    public static void main(String[] args) throws Exception {
        String method = null, pa = null, pb = null, out = null;
        int n = -1, p = 1, warmup = 0, reps = 1, batch = 0;
        double targetMs = 100;
        for (int i = 0; i + 1 < args.length; i += 2) {
            switch (args[i]) {
                case "--method" -> method = args[i + 1];
                case "--n" -> n = Integer.parseInt(args[i + 1]);
                case "--threads" -> p = Integer.parseInt(args[i + 1]);
                case "--a" -> pa = args[i + 1];
                case "--b" -> pb = args[i + 1];
                case "--warmup" -> warmup = Integer.parseInt(args[i + 1]);
                case "--reps" -> reps = Integer.parseInt(args[i + 1]);
                case "--batch" -> batch = Integer.parseInt(args[i + 1]);
                case "--target-ms" -> targetMs = Double.parseDouble(args[i + 1]);
                case "--out" -> out = args[i + 1];
                default -> throw new IllegalArgumentException("unknown option " + args[i]);
            }
        }
        if (method == null || n <= 0 || p <= 0 || pa == null || pb == null) {
            System.err.println("usage: MatMul3 --method seq|seq_vapi|static|static_vapi|stream --n N "
                    + "--threads P --a A.bin --b B.bin [--warmup W] [--reps R] [--batch K|0=auto] "
                    + "[--target-ms 100] [--out C.bin]");
            System.exit(1);
        }
        if ((long) n * n > Integer.MAX_VALUE) throw new IllegalArgumentException("n too large for a flat double[]");
        if (method.startsWith("seq") && p != 1) throw new IllegalArgumentException("sequential methods use --threads 1");

        // ---- 1. load and set up (not timed)
        double[] A = loadDense(pa, n), B = loadDense(pb, n);
        double[] C = new double[n * n];
        try (Kernel kernel = make(method, A, B, C, n, p)) {
            System.out.println("#threads_requested," + p);
            System.out.println("#available_processors," + Runtime.getRuntime().availableProcessors());
            if (method.endsWith("_vapi")) System.out.println("#vector_lanes," + VecKernels.lanes());

            // ---- 2. warm-up (each round >= targetMs so the JIT compiles the kernel)
            double lastWarmupPerMult = -1;
            for (int r = 0; r < warmup; r++) {
                long c0 = cpuNs(), t0 = System.nanoTime(), t1;
                int count = 0;
                do {
                    kernel.run(1);
                    count++;
                    t1 = System.nanoTime();
                } while (t1 - t0 < targetMs * 1e6);
                long c1 = cpuNs();
                lastWarmupPerMult = (double) (t1 - t0) / count;
                System.out.println("warmup," + r + "," + (t1 - t0) + "," + count + "," + sum(C) + "," + (c1 - c0));
            }

            // ---- 3. batch size
            if (batch <= 0 && lastWarmupPerMult > 0) {
                batch = (int) Math.max(1, Math.ceil(targetMs * 1e6 / lastWarmupPerMult));
            } else if (batch <= 0) {
                long c0 = cpuNs(), t0 = System.nanoTime();
                kernel.run(1);
                long t1 = System.nanoTime(), c1 = cpuNs();
                long t = Math.max(1, t1 - t0);
                batch = (int) Math.max(1, Math.ceil(targetMs * 1e6 / t));
                System.out.println("calibration,0," + t + ",1," + sum(C) + "," + (c1 - c0));
            }

            // ---- 4. measured repetitions
            for (int r = 0; r < reps; r++) {
                long c0 = cpuNs(), t0 = System.nanoTime();
                kernel.run(batch);
                long t1 = System.nanoTime(), c1 = cpuNs();
                System.out.println("measured," + r + "," + (t1 - t0) + "," + batch + "," + sum(C) + "," + (c1 - c0));
            }

            // ---- 5. which threads actually worked (one untimed multiplication)
            kernel.track(true);
            kernel.run(1);
            kernel.track(false);
            System.out.println("#distinct_worker_threads," + kernel.distinctThreads());
            System.out.println("#tasks_per_multiplication," + kernel.tasks());
        }

        if (out != null) saveDense(out, C);
        Runtime rt = Runtime.getRuntime();
        System.out.println("#array_storage_bytes," + 3L * 8 * n * n);
        System.out.println("#java_heap_used_bytes," + (rt.totalMemory() - rt.freeMemory()));
        System.out.println("#jvm_peak_thread_count," + ManagementFactory.getThreadMXBean().getPeakThreadCount());
    }
}
