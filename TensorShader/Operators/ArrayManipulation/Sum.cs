using System;
using System.Collections.Generic;
using System.Linq;

namespace TensorShader.Operators.ArrayManipulation {
    /// <summary>テンソル総和</summary>
    internal class Sum : Operator {
        /// <summary>形状</summary>
        public Shape Shape { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Sum(Shape shape, int num) {
            if (num < 1) {
                throw new ArgumentException(nameof(num));
            }

            this.arguments = new List<(ArgumentType type, Shape shape)>();

            for (int i = 0; i < num; i++) {
                this.arguments.Add((ArgumentType.In, shape));
            }

            this.arguments.Add((ArgumentType.Out, shape));

            this.Shape = shape;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            float[][] inmaps = tensors.Take(tensors.Length - 1).Select((tensor)=>tensor.Buffer).ToArray();
            float[] refmap = tensors.Last().Buffer;

            TensorShaderAvxBackend.Elementwise.Sum(0, (uint)Shape.Length, inmaps, refmap);
        }
    }
}
