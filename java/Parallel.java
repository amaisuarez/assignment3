import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Future;
import java.util.stream.IntStream;

/**
 * Shared-memory parallel versions of the i-k-j kernel. Both split the rows of C
 * between threads; they differ in how the work is divided and scheduled.
 *
 *  StaticRows  java.util.concurrent fixed thread pool with p threads. The n rows
 *              are cut into p contiguous blocks whose sizes differ by at most one
 *              row (static block partition, like OpenMP schedule(static)). One task
 *              per block; the caller waits for all of them (invokeAll).
 *  StreamRows  ForkJoinPool with parallelism p running IntStream.range(0, n)
 *              .parallel(): the framework splits the row range recursively and
 *              balances it by work stealing (dynamic scheduling, one row per leaf
 *              call).
 *
 * Safety. Each row of C is written by exactly one task; A and B are only read.
 * The pools are created before timing and reused for every multiplication.
 * Completion of a task happens-before the return of Future.get() / invokeAll(),
 * so the calling thread sees the whole of C after run() returns.
 *
 * Thread counting. If track(true) is set, every task records the name of the
 * thread that ran it. The driver does this in one extra, untimed multiplication
 * to report how many distinct worker threads really took part. This is at most p;
 * it can be lower for small n, because a worker that finishes its task early may
 * take the next queued task before an idle worker does.
 */
final class Parallel {

    private Parallel() {}

    /** Row kernel to run: scalar i-k-j or the Vector API version. */
    interface RowKernel {
        void rows(double[] A, double[] B, double[] C, int n, int lo, int hi);
    }

    static RowKernel rowKernel(boolean vapi) {
        return vapi ? VecKernels::ikjRows : Kernels::ikjRows;
    }

    /** Contiguous block partition of n rows into p parts: bounds[t]..bounds[t+1]. */
    static int[] blockBounds(int n, int p) {
        int[] b = new int[p + 1];
        for (int t = 0; t <= p; t++) b[t] = (int) ((long) n * t / p);
        return b;
    }

    abstract static class Runner implements AutoCloseable {
        final double[] A, B, C;
        final int n, p;
        final RowKernel kernel;
        final Set<String> seen = ConcurrentHashMap.newKeySet();
        volatile boolean track = false;

        Runner(double[] A, double[] B, double[] C, int n, int p, RowKernel kernel) {
            this.A = A; this.B = B; this.C = C; this.n = n; this.p = p; this.kernel = kernel;
        }

        void track(boolean on) {
            if (on) seen.clear();
            track = on;
        }
        int distinctThreads() { return seen.size(); }
        /** Tasks per multiplication (static partition), or the n row tasks of the stream. */
        abstract int tasks();
        void note() { if (track) seen.add(Thread.currentThread().getName()); }

        abstract void multiply() throws InterruptedException, ExecutionException;

        @Override public abstract void close();
    }

    static final class StaticRows extends Runner {
        final ExecutorService pool;
        final List<Callable<Void>> tasks = new ArrayList<>();

        StaticRows(double[] A, double[] B, double[] C, int n, int p, RowKernel kernel) {
            super(A, B, C, n, p, kernel);
            pool = Executors.newFixedThreadPool(p);
            int[] b = blockBounds(n, p);
            for (int t = 0; t < p; t++) {
                final int lo = b[t], hi = b[t + 1];
                if (lo == hi) continue;                     // n < p: no empty tasks
                tasks.add(() -> {
                    note();
                    kernel.rows(this.A, this.B, this.C, this.n, lo, hi);
                    return null;
                });
            }
        }

        void multiply() throws InterruptedException, ExecutionException {
            for (Future<Void> f : pool.invokeAll(tasks)) f.get();   // rethrows task failures
        }

        int tasks() { return tasks.size(); }

        public void close() { pool.shutdownNow(); }
    }

    static final class StreamRows extends Runner {
        final ForkJoinPool pool;

        StreamRows(double[] A, double[] B, double[] C, int n, int p, RowKernel kernel) {
            super(A, B, C, n, p, kernel);
            pool = new ForkJoinPool(p);
        }

        void multiply() throws InterruptedException, ExecutionException {
            // A parallel stream started from inside a ForkJoinPool task runs in that
            // pool, not in the common pool, so exactly p workers are available.
            pool.submit(() -> IntStream.range(0, n).parallel().forEach(i -> {
                note();
                kernel.rows(A, B, C, n, i, i + 1);
            })).get();
        }

        int tasks() { return n; }

        public void close() { pool.shutdownNow(); }
    }
}
