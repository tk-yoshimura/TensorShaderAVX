namespace TensorShader.Operators.FactorArithmetric {
    /// <summary>符号付きべき乗</summary>
    internal class SignedPow : FactorArithmetric {
        /// <summary>コンストラクタ</summary>
        public SignedPow(Shape shape)
            : base(shape){ }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outmap = tensors[2];

            TensorShaderAvxBackend.Elementwise.SignedPowConstant((uint)Shape.Length, inmap2.Buffer[0], inmap1.Buffer, outmap.Buffer);
        }
    }
}
