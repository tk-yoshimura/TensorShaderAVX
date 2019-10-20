namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>符号反転</summary>
    internal class Neg : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Neg(Shape shape)
            : base(shape){ }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderAvxBackend.Elementwise.Neg(0, (uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
