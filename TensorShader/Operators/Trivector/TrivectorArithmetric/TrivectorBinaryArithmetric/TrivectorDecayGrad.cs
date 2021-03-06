namespace TensorShader.Operators.TrivectorBinaryArithmetric {
    /// <summary>3次元ベクトルDecay勾配</summary>
    internal class TrivectorDecayGrad : TrivectorBinaryArithmetric {
        /// <summary>コンストラクタ</summary>
        public TrivectorDecayGrad(Shape shape)
            : base(shape) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outmap = tensors[2];

            TensorShaderAvxBackend.Trivector.DecayGrad((uint)Shape.Length, inmap1.Buffer, inmap2.Buffer, outmap.Buffer);
        }
    }
}
