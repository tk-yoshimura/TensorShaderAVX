namespace TensorShader.Operators.QuaternionUnaryArithmetric {
    /// <summary>四元数Decay</summary>
    internal class QuaternionDecay : QuaternionUnaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public QuaternionDecay(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderAvxBackend.Quaternion.Decay(0, (uint)Shape.Length, inmap.Buffer, outmap.Buffer);
        }
    }
}
