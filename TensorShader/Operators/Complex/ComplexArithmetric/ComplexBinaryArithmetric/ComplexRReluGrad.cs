namespace TensorShader.Operators.ComplexBinaryArithmetric {
    /// <summary>複素RRelu勾配</summary>
    internal class ComplexRReluGrad : ComplexBinaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public ComplexRReluGrad(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outmap = tensors[2];

            TensorShaderAvxBackend.Complex.RReluGrad(0, (uint)Shape.Length, inmap1.Buffer, inmap2.Buffer, outmap.Buffer);
        }
    }
}
