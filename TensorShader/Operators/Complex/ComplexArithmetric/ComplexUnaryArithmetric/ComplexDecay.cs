namespace TensorShader.Operators.ComplexUnaryArithmetric {
    /// <summary>複素Decay</summary>
    internal class ComplexDecay : ComplexUnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public ComplexDecay(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderAvxBackend.Complex.Decay(0, (uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
