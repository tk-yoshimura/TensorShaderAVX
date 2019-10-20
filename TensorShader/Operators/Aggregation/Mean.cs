namespace TensorShader.Operators.Aggregation {
    /// <summary>平均</summary>
    internal class Mean : Aggregation {
        /// <summary>コンストラクタ</summary>
        public Mean(Shape shape, int axis)
            : base(shape, axis){ }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            TensorShaderAvxBackend.Aggregation.Average(0, AxisLength * Stride, inmap.Buffer, 0, Stride, outmap.Buffer, Slides);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor outmap) {
            Execute(new Tensor[] { inmap, outmap });
        }
    }
}
