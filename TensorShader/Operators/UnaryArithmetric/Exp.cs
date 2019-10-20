namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>指数関数</summary>
    internal class Exp : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Exp(Shape shape)
            : base(shape){ }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderAvxBackend.Elementwise.Exp(0, (uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
