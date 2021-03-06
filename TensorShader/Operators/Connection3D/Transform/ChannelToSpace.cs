using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace TensorShader.Operators.Connection3D {
    /// <summary>チャネル次元を空間方向に展開</summary>
    internal class ChannelToSpace : Operator {
        /// <summary>入力チャネル数</summary>
        public int InChannels { private set; get; }

        /// <summary>出力チャネル数</summary>
        public int OutChannels { private set; get; }

        /// <summary>倍率</summary>
        public int Scale { private set; get; }

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        public ChannelToSpace(int inwidth, int inheight, int indepth, int inchannels, int scale, int batch = 1) {
            if (scale < 2 || inchannels % (scale * scale * scale) != 0) {
                throw new ArgumentException($"{nameof(scale)}, {nameof(inchannels)}");
            }

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Map3D(inchannels, inwidth, inheight, indepth, batch)),
                (ArgumentType.Out, Shape.Map3D(inchannels / checked(scale * scale * scale), checked(inwidth * scale), checked(inheight * scale), checked(indepth * scale), batch)),
            };

            this.InChannels = inchannels;
            this.OutChannels = inchannels / (scale * scale * scale);

            this.Scale = scale;
            this.Batch = batch;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            Parallel.For(0, Batch, (th) => {
                TensorShaderAvxBackend.Transform.ChannelToSpace3D((uint)OutChannels, (uint)inmap.Width, (uint)inmap.Height, (uint)inmap.Depth, (uint)Batch, (uint)th, (uint)Scale, inmap.Buffer, outmap.Buffer);
            });
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor outmap) {
            Execute(new Tensor[] { inmap, outmap });
        }
    }
}
