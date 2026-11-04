import java.util.Arrays;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Optional vectorisation extension: the i-k-j row kernel written with explicit
 * SIMD instructions through the Java Vector API (incubator module
 * jdk.incubator.vector; sixth incubator in JDK 21, JEP 448).
 *
 * The inner j loop is processed L doubles at a time, where L is the lane count of
 * the preferred species of the CPU (4 doubles = 256 bits with AVX2). For each lane
 * it computes c = c + a * b with a separate multiply and add (no fma), i.e. the
 * same two IEEE operations per element as the scalar kernel, in the same k order.
 * Vectorising over j only changes how many independent output elements are
 * updated per instruction, never the order of the additions into one element, so
 * the result is bit-identical to Kernels.ikj. The last n mod L columns are done
 * by a scalar tail loop.
 */
final class VecKernels {

    static final VectorSpecies<Double> S = DoubleVector.SPECIES_PREFERRED;

    private VecKernels() {}

    static int lanes() { return S.length(); }

    static void ikjRows(double[] A, double[] B, double[] C, int n, int lo, int hi) {
        Arrays.fill(C, lo * n, hi * n, 0.0);
        int upper = S.loopBound(n);
        for (int i = lo; i < hi; i++) {
            int ri = i * n;
            for (int k = 0; k < n; k++) {
                double a = A[ri + k];
                DoubleVector va = DoubleVector.broadcast(S, a);
                int rk = k * n;
                int j = 0;
                for (; j < upper; j += S.length()) {
                    DoubleVector vb = DoubleVector.fromArray(S, B, rk + j);
                    DoubleVector vc = DoubleVector.fromArray(S, C, ri + j);
                    vc.add(va.mul(vb)).intoArray(C, ri + j);
                }
                for (; j < n; j++)
                    C[ri + j] += a * B[rk + j];
            }
        }
    }
}
