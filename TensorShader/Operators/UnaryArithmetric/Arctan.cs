namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>逆正接関数</summary>
    internal class Arctan : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Arctan(Shape shape)
            : base(shape){ }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderAvxBackend.Elementwise.ArcTan((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
