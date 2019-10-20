namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>2乗平均</summary>
        public static VariableNode SquareMean(VariableNode x, int[] axes = null, bool keepdims = false) {
            return Mean(Square(x), axes, keepdims);
        }
    }

    public partial class Tensor {
        /// <summary>2乗平均</summary>
        public static Tensor SquareMean(Tensor x, int[] axes = null, bool keepdims = false) {
            return Mean(Square(x), axes, keepdims);
        }
    }
}
