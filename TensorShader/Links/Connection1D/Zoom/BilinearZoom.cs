using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>1次元バイリニア補間</summary>
        /// <remarks>倍率2固定</remarks>
        public static Field BilinearZoom1D(Field x) {
            Field y = new Field();
            Link link = new Links.Connection1D.BilinearZoom(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Connection1D {
    /// <summary>1次元バイリニア補間</summary>
    /// <remarks>倍率2固定</remarks>
    public class BilinearZoom : Link {
        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>出力項</summary>
        protected Field Y => OutFields[0];

        /// <summary>コンストラクタ</summary>
        public BilinearZoom(Field infield, Field outfield)
            : base(new Field[]{ infield }, new Field[]{ outfield }) {
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(BilinearZoom1D(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(AveragePooling1D(Y.Grad, stride:2));
            }
        }
    }
}
