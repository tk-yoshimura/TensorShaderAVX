using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>3次元ベクトル1次元畳み込み</summary>
        public static Field TrivectorConvolution1D(Field x, Field w, int stride) {
            Field y = new Field();
            Link link = new Links.TrivectorConvolution.TrivectorConvolution1D(x, w, y, stride);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.TrivectorConvolution {
    /// <summary>3次元ベクトル1次元畳み込み</summary>
    public class TrivectorConvolution1D : Link {
        /// <summary>ストライド</summary>
        public int Stride { private set; get; }

        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>カーネル項</summary>
        protected Field W => InFields[1];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public TrivectorConvolution1D(Field infield, Field kernelfield, Field outfield, int stride)
            : base(new Field[]{ infield, kernelfield }, outfield) {
            this.Stride = stride;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(TrivectorConvolution1D(X.Value, W.Value, Stride, gradmode: false));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(TrivectorDeconvolution1D(Y.Grad, W.Value, Stride, gradmode: true, X.Shape));
            }

            if (W.EnableBackprop) {
                W.AddGrad(TrivectorKernelProduct1D(X.Value, Y.Grad, W.Value, W.Shape.Width, Stride, transpose: false));
            }
        }
    }
}
