namespace TensorShader.Operators.BinaryArithmetric {
    /// <summary>符号付きべき乗</summary>
    internal class SignedPow : BinaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public SignedPow(Shape shape)
            : base(shape){ }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outmap = tensors[2];

            TensorShaderAvxBackend.Elementwise.SignedPow(0, (uint)Shape.Length, inmap1.Buffer, inmap2.Buffer, outmap.Buffer);
        }
    }

    /// <summary>符号付きべき乗</summary>
    internal class SignedPowRightConstant : BinaryRightConstantArithmetric {
        /// <summary>コンストラクタ</summary>
        public SignedPowRightConstant(float c, Shape shape)
            : base(c, shape){ }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderAvxBackend.Elementwise.SignedPowConstant(0, (uint)Shape.Length, Constant, inmap.Buffer, outmap.Buffer);
        }
    }
}
