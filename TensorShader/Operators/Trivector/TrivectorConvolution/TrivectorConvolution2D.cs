using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace TensorShader.Operators.TrivectorConvolution {
    /// <summary>3次元ベクトル2次元畳み込み</summary>
    internal class TrivectorConvolution2D : Operator {
        /// <summary>入力チャネル</summary>
        public int InChannels { private set; get; }

        /// <summary>出力チャネル</summary>
        public int OutChannels { private set; get; }

        /// <summary>フィルタサイズ</summary>
        /// <remarks>奇数を指定すること</remarks>
        public int KernelWidth { private set; get; }

        /// <summary>フィルタサイズ</summary>
        /// <remarks>奇数を指定すること</remarks>
        public int KernelHeight { private set; get; }

        /// <summary>ストライド</summary>
        public int Stride { private set; get; }

        /// <summary>勾配</summary>
        public bool GradMode { private set; get; }

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        public TrivectorConvolution2D(int inwidth, int inheight, int inchannels, int outchannels, int kwidth, int kheight, int stride, bool gradmode = false, int batch = 1) {
            if (stride < 1) {
                throw new ArgumentException(nameof(stride));
            }

            if (inchannels % 3 != 0) {
                throw new ArgumentException(ExceptionMessage.ArgumentMultiple(nameof(inchannels), inchannels, 3));
            }
            if (outchannels % 3 != 0) {
                throw new ArgumentException(ExceptionMessage.ArgumentMultiple(nameof(outchannels), outchannels, 3));
            }

            int outwidth = (inwidth - kwidth) / stride + 1;
            int outheight = (inheight - kheight) / stride + 1;

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Map2D(inchannels, inwidth, inheight, batch)),
                (ArgumentType.In, Shape.Kernel2D(inchannels / 3 * 4, outchannels / 3, kwidth, kheight)),
                (ArgumentType.Out, Shape.Map2D(outchannels, outwidth, outheight, batch)),
            };

            this.InChannels = inchannels;
            this.OutChannels = outchannels;
            this.KernelWidth = kwidth;
            this.KernelHeight = kheight;
            this.Stride = stride;
            this.GradMode = gradmode;
            this.Batch = batch;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], infilter = tensors[1], outmap = tensors[2];

            Parallel.For(0, Batch, (th) => {
                TensorShaderAvxBackend.Trivector.Convolution2D((uint)InChannels, (uint)OutChannels, 
                                                               (uint)inmap.Width, (uint)inmap.Height, 
                                                               (uint)Batch, (uint)th, 
                                                               (uint)KernelWidth, (uint)KernelHeight, 
                                                               (uint)Stride, GradMode, 
                                                               inmap.Buffer, infilter.Buffer, outmap.Buffer);
            });
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap, Tensor infilter, Tensor outmap) {
            Execute(new Tensor[]{ inmap, infilter, outmap });
        }
    }
}
