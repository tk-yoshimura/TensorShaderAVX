using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>3次元バイリニア補間</summary>
        /// <remarks>倍率2固定</remarks>
        public static Field BilinearZoom3D(Field x) {
            Field y = new Field();
            Link link = new Links.Connection3D.BilinearZoom(x, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Connection3D {
    /// <summary>3次元バイリニア補間</summary>
    /// <remarks>倍率2固定</remarks>
    public class BilinearZoom : Link {
        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>出力項</summary>
        protected Field Y => OutField;

        /// <summary>コンストラクタ</summary>
        public BilinearZoom(Field infield, Field outfield)
            : base(new Field[]{ infield }, outfield) {
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(BilinearZoom3D(X.Value));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(AveragePooling3D(Y.Grad, stride:2));
            }
        }
    }
}
