using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>複素3次元畳み込み</summary>
        public static Field ComplexConvolution3D(Field x, Field w, int stride) {
            Field y = new Field();
            Link link = new Links.ComplexConvolution.ComplexConvolution3D(x, w, y, stride);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.ComplexConvolution {
    /// <summary>複素3次元畳み込み</summary>
    public class ComplexConvolution3D : Link {
        /// <summary>ストライド</summary>
        public int Stride { private set; get; }

        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>カーネル項</summary>
        protected Field W => InFields[1];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public ComplexConvolution3D(Field infield, Field kernelfield, Field outfield, int stride)
            : base(new Field[]{ infield, kernelfield }, outfield) {
            this.Stride = stride;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(ComplexConvolution3D(X.Value, W.Value, Stride, gradmode: false));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(ComplexDeconvolution3D(Y.Grad, W.Value, Stride, gradmode: true, X.Shape));
            }

            if (W.EnableBackprop) {
                W.AddGrad(ComplexKernelProduct3D(X.Value, Y.Grad, W.Shape.Width, W.Shape.Height, W.Shape.Depth, Stride, transpose: false));
            }
        }
    }
}
