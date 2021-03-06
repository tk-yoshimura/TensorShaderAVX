namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>複素Decay勾配</summary>
        public static VariableNode ComplexDecayGrad(VariableNode x1, VariableNode x2) {
            return ComplexBinaryArithmetric(x1, x2, new Operators.ComplexBinaryArithmetric.ComplexDecayGrad(x1.Shape));
        }
    }

    public partial class Tensor {
        /// <summary>複素Decay勾配</summary>
        public static Tensor ComplexDecayGrad(Tensor x1, Tensor x2) {
            return ComplexBinaryArithmetric(x1, x2, new Operators.ComplexBinaryArithmetric.ComplexDecayGrad(x1.Shape));
        }
    }
}
