namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>分散</summary>
        public static VariableNode Variance(VariableNode x, int[] axes = null, bool keepdims = false) {
            return SquareMean(x, axes, keepdims) - Square(Mean(x, axes, keepdims));
        }
    }

    public partial class Tensor {
        /// <summary>分散</summary>
        public static Tensor Variance(Tensor x, int[] axes = null, bool keepdims = false) {
            return SquareMean(x, axes, keepdims) - Square(Mean(x, axes, keepdims));
        }
    }
}
