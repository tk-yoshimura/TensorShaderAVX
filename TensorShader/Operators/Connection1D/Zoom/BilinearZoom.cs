using System;
using System.Threading.Tasks;

namespace TensorShader.Operators.Connection1D {
    /// <summary>バイリニア補間</summary>
    /// <remarks>倍率2固定</remarks>
    internal class BilinearZoom : Zoom{
        /// <summary>コンストラクタ</summary>
        public BilinearZoom(int inwidth, int channels, int batch = 1)
            : base(inwidth, channels, scale: 2, batch) { }

        /// <summary>操作を実行</summary>
        public override void Execute(params Tensor[] tensors) {
            CheckArgumentShapes(tensors);

            Tensor inmap = tensors[0], outmap = tensors[1];

            Parallel.For(0, Batch, (th) => {
                TensorShaderAvxBackend.Zoom.LinearZoom1D((uint)Channels, (uint)inmap.Width, 
                                                         (uint)Batch, (uint)th, inmap.Buffer, outmap.Buffer);
            });
        }
    }
}
