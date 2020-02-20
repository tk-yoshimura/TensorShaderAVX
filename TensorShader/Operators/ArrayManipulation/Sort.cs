using System.Collections.Generic;
using System.Threading.Tasks;

namespace TensorShader.Operators.ArrayManipulation {
    /// <summary>ソート</summary>
    internal class Sort : Operator {
        /// <summary>形状</summary>
        public Shape Shape { private set; get; }

        /// <summary>ストライド</summary>
        public int Stride { private set; get; }

        /// <summary>軸長さ</summary>
        public int AxisLength { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Sort(Shape shape, int axis) {
            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, shape),
                (ArgumentType.Out, shape),
            };

            int stride = 1;
            for (int i = 0; i < axis; i++) {
                stride *= shape[i];
            }

            this.Shape = shape;
            this.Stride = stride;
            this.AxisLength = shape[axis];
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor intensor = tensors[0], outtensor = tensors[1];

            TensorShaderAvxBackend.ArrayManipulation.Sort((uint)Stride, (uint)AxisLength, (uint)(outtensor.Length / (Stride * AxisLength)), intensor.Buffer, outtensor.Buffer);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor intensor, Tensor outtensor) {
            Execute(new Tensor[] { intensor, outtensor });
        }
    }
}
