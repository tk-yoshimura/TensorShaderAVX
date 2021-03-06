using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace TensorShader.Operators.Connection1D {
    /// <summary>チャネルごとの1次元畳み込み</summary>
    internal class ChannelwiseConvolution : Operator {
        /// <summary>チャネル</summary>
        public int Channels { private set; get; }

        /// <summary>フィルタサイズ</summary>
        /// <remarks>奇数を指定すること</remarks>
        public int KernelWidth { private set; get; }

        /// <summary>ストライド</summary>
        public int Stride { private set; get; }

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        public ChannelwiseConvolution(int inwidth, int channels, int kwidth, int stride, int batch = 1){
            if (stride < 1) {
                throw new ArgumentException(nameof(stride));
            }

            int outwidth = (inwidth - kwidth) / stride + 1;

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Map1D(channels, inwidth, batch)),
                (ArgumentType.In, Shape.Kernel1D(channels, 1, kwidth)),
                (ArgumentType.Out, Shape.Map1D(channels, outwidth, batch)),
            };

            this.Channels = channels;
            this.KernelWidth = kwidth;
            this.Stride = stride;
            this.Batch = batch;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], infilter = tensors[1], outmap = tensors[2];

            Parallel.For(0, Batch, (th) => {
                TensorShaderAvxBackend.Convolution.ChannelwiseConvolution1D((uint)Channels,
                                                                      (uint)inmap.Width, (uint)Batch, (uint)th,
                                                                      (uint)KernelWidth, (uint)Stride,
                                                                      inmap.Buffer, infilter.Buffer, outmap.Buffer);
            });
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor infilter, Tensor outmap) {
            Execute(new Tensor[]{ inmap, infilter, outmap });
        }
    }
}
