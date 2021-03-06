using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace TensorShader.Operators.ComplexConvolution {
    /// <summary>複素カーネル積</summary>
    internal class ComplexKernelProduct3D : Operator{
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

        /// <summary>フィルタサイズ</summary>
        /// <remarks>奇数を指定すること</remarks>
        public int KernelDepth { private set; get; }

        /// <summary>ストライド</summary>
        public int Stride { private set; get; }

        /// <summary>転置</summary>
        public bool Transpose { private set; get; }

        /// <summary>バッチサイズ</summary>
        public int Batch { private set; get; }

        /// <summary>コンストラクタ</summary>
        public ComplexKernelProduct3D(int inwidth, int inheight, int indepth, int inchannels, int outchannels, int kwidth, int kheight, int kdepth, int stride, bool transpose = false, int batch = 1) {
            if (stride < 1) {
                throw new ArgumentException(nameof(stride));
            }

            if (inchannels % 2 != 0) {
                throw new ArgumentException(ExceptionMessage.ArgumentMultiple(nameof(inchannels), inchannels, 2));
            }
            if (outchannels % 2 != 0) {
                throw new ArgumentException(ExceptionMessage.ArgumentMultiple(nameof(outchannels), outchannels, 2));
            }

            int outwidth = (inwidth - kwidth) / stride + 1;
            int outheight = (inheight - kheight) / stride + 1;
            int outdepth = (indepth - kdepth) / stride + 1;

            this.arguments = new List<(ArgumentType type, Shape shape)>{
                (ArgumentType.In, Shape.Map3D(inchannels, inwidth, inheight, indepth, batch)),
                (ArgumentType.In, Shape.Map3D(outchannels, outwidth, outheight, outdepth, batch)),
                (ArgumentType.Out, Shape.Kernel3D(inchannels, outchannels / 2, kwidth, kheight, kdepth)),
            };

            this.InChannels = inchannels;
            this.OutChannels = outchannels;
            this.KernelWidth = kwidth;
            this.KernelHeight = kheight;
            this.KernelDepth = kdepth;
            this.Stride = stride;
            this.Transpose = transpose;
            this.Batch = batch;
        }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap1 = tensors[0], inmap2 = tensors[1], outfilter = tensors[2];

            Parallel.For(0, OutChannels / 2, (outch) => {
                TensorShaderAvxBackend.Complex.KernelProduct3D((uint)InChannels, (uint)OutChannels, 
                                                               (uint)inmap1.Width, (uint)inmap1.Height, (uint)inmap1.Depth, 
                                                               (uint)Batch, (uint)outch * 2, 
                                                               (uint)KernelWidth, (uint)KernelHeight, (uint)KernelDepth, 
                                                               (uint)Stride, Transpose, 
                                                               inmap1.Buffer, inmap2.Buffer, outfilter.Buffer);
            });
        }

        /// <summary>操作を実行</summary>
        public void Execute(Tensor inmap1, Tensor inmap2, Tensor outfilter) {
            Execute(new Tensor[]{ inmap1, inmap2, outfilter });
        }
    }
}
