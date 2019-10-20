namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>平均/分散</summary>
        public static (VariableNode mean, VariableNode variance) MeanVariance(VariableNode x, int[] axes = null, bool keepdims = false) {
            VariableNode mean = Mean(x, axes, keepdims);
            VariableNode variance = SquareMean(x, axes, keepdims) - Square(mean);

            return (mean, variance);
        }
    }

    public partial class Tensor {
        /// <summary>平均/分散</summary>
        public static (Tensor mean, Tensor variance) MeanVariance(Tensor x, int[] axes = null, bool keepdims = false) {
            Tensor mean = Mean(x, axes, keepdims);
            Tensor variance = SquareMean(x, axes, keepdims) - Square(mean);

            return (mean, variance);
        }
    }
}
