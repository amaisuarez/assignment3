import java.util.Arrays;

/**
 * Dense n x n matrix multiplication kernels, float64, row-major double[]
 * (element (i,j) at i*n + j), the same storage as Assignments 1 and 2.
 *
 * All kernels here use the i-k-j loop order. For every output element C[i][j]
 * the products A[i][k]*B[k][j] are added for k = 0, 1, ..., n-1 in increasing
 * order, starting from 0.0. Java does not reassociate floating-point additions
 * or fuse them into FMA (strict IEEE 754 since Java 17, JEP 306), so every kernel
 * in this assignment, sequential or parallel, returns bit-identical results.
 * tests/validate.py checks this.
 */
final class Kernels {

    private Kernels() {}

    /**
     * Sequential baseline: the i-k-j kernel of Assignment 2, copied unchanged from
     * java/Dense.java, method ikj (repo amaisuarez/assignment2). It was the fastest
     * sequential method of Assignment 2 (cache blocking gave no consistent gain on
     * top of it), so it is the equivalent sequential counterpart of the parallel
     * versions below.
     */
    static void ikj(double[] A, double[] B, double[] C, int n) {
        Arrays.fill(C, 0.0);
        for (int i = 0; i < n; i++) {
            int ri = i * n;
            for (int k = 0; k < n; k++) {
                double a = A[ri + k];
                int rk = k * n;
                for (int j = 0; j < n; j++)
                    C[ri + j] += a * B[rk + j];
            }
        }
    }

    /**
     * The same i-k-j kernel restricted to the rows lo..hi-1 of C. This is the unit
     * of work of the parallel versions: a thread that owns rows [lo, hi) zeroes and
     * writes only those rows of C and only reads A and B, so no two threads ever
     * write the same memory location and no locking is needed.
     * ikjRows(A, B, C, n, 0, n) does exactly the same work as ikj(A, B, C, n).
     */
    static void ikjRows(double[] A, double[] B, double[] C, int n, int lo, int hi) {
        Arrays.fill(C, lo * n, hi * n, 0.0);
        for (int i = lo; i < hi; i++) {
            int ri = i * n;
            for (int k = 0; k < n; k++) {
                double a = A[ri + k];
                int rk = k * n;
                for (int j = 0; j < n; j++)
                    C[ri + j] += a * B[rk + j];
            }
        }
    }
}
