import org.apache.commons.math3.linear.*;

import java.util.ArrayList;
import java.util.List;

/**
 * RBF: f(x) = sum_i [lambda_i * phi(||x - x_i||)]
 *       phi(r) = r^2 log r
 *       phi(r) = sqrt(r^2 + c^2)
 *       phi(r) = exp(-cr^2)
 *       phi(r) = r
 *       phi(r) = r^3 
 * 今回はr^3を使用する
 */
public class RBF{

    // RBF中心 (オンサーフェス点) の座標リスト
    private final List<double[]> centers = new ArrayList<>();
    // 各中心に対応する係数(lambda)
    private double[] lambdas;

    // 基準データ (サンプル点) - f(x_i) = value_i
    // オンサーフェス点(0) + オフサーフェス点(±1など)
    public static class SamplePoint {
        public double x, y, z;
        public double value;

        public SamplePoint(double x, double y, double z, double value) {
            this.x = x;
            this.y = y;
            this.z = z;
            this.value = value;
        }
    }

    /**
     * RBF を構築するための点群(オンサーフェス+オフサーフェス)を渡し、
     * 連立方程式を解いて lambda を求める。
     * ここでは「オンサーフェス点をそのままRBFの中心」と仮定。
     */
    public void buildRBF(List<SamplePoint> allPoints) {
        // 1) centers の抽出: value = 0 の点のみを RBF中心とする
        for (SamplePoint sp : allPoints) {
            if (Math.abs(sp.value) < 1e-9) {
                centers.add(new double[]{sp.x, sp.y, sp.z});
            }
        }
        int N = allPoints.size();
        int K = centers.size();

        // 2) 行列 A (N x K), 右辺 b (N)
        double[][] matA = new double[N][K];
        double[]   vecB = new double[N];

        for (int i = 0; i < N; i++) {
            SamplePoint sp = allPoints.get(i);
            vecB[i] = sp.value;

            for (int j = 0; j < K; j++) {
                double[] c = centers.get(j);
                double dx = sp.x - c[0];
                double dy = sp.y - c[1];
                double dz = sp.z - c[2];
                double r  = Math.sqrt(dx * dx + dy * dy + dz * dz);
                matA[i][j] = r * r * r; // phi(r) = r^3
            }
        }

        // 連立方程式を解く
        RealMatrix A = MatrixUtils.createRealMatrix(matA);
        RealVector b = MatrixUtils.createRealVector(vecB);

        DecompositionSolver solver = new SingularValueDecomposition(A).getSolver();
        RealVector sol = solver.solve(b);
        lambdas = sol.toArray();
        System.out.println("Complete.");
    }

    /**
     * f(x, y, z) を評価
     */
    public double evaluate(double x, double y, double z) {
        if (lambdas == null) {
            throw new IllegalStateException("RBF not built yet!");
        }
        double val = 0.0;
        for (int j = 0; j < centers.size(); j++) {
            double[] c = centers.get(j);
            double dx = x - c[0];
            double dy = y - c[1];
            double dz = z - c[2];
            double r  = Math.sqrt(dx * dx + dy * dy + dz * dz);
            val += lambdas[j] * (r*r*r);
        }
        return val;
    }
}

