using System;
using System.Collections.Generic;

namespace TensorShader.Operators.Connection3D {
    /// <summary>ストライドプーリング</summary>
    internal class StridePooling : Operator {
        /// <summary>チャネル数</summary>
        public int Channels { private set; get; }

        /// <summary>ストライド</summary>
        public int Stride { private set; get; }

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        public StridePooling(int inwidth, int inheight, int indepth, int channels, int stride, int batch = 1) {
            if (stride < 2) {
                throw new ArgumentException(nameof(stride));
            }

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Map3D(channels, inwidth, inheight, indepth, batch)),
                (ArgumentType.Out, Shape.Map3D(channels, inwidth / stride, inheight / stride, indepth / stride, batch)),
            };

            this.Channels = channels;
            this.Stride = stride;
            this.Batch = batch;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            throw new NotImplementedException();

            //TensorShaderAvxBackend.Pool.StridePool3D((uint)Channels, (uint)inmap.Width, (uint)inmap.Height, (uint)inmap.Depth, (uint)Batch, (uint)Stride, inmap.Buffer, outmap.Buffer);
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor outmap) {
            Execute(new Tensor[] { inmap, outmap });
        }
    }
}
