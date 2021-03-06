namespace TensorShader.Operators.UnaryArithmetric {
    /// <summary>Relu</summary>
    internal class Relu : UnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public Relu(Shape shape)
            : base(shape){ }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderAvxBackend.Elementwise.Relu((uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
