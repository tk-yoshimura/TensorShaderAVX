namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>双曲線余弦関数</summary>
    internal class Cosh : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Cosh(Shape shape)
            : base(shape){ }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderAvxBackend.Elementwise.Cosh((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
