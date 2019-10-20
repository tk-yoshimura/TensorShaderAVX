namespace TensorShader {
    public partial class Field {
        /// <summary>正負</summary>
        public static (Field y_pos, Field y_neg) PosNeg(Field x) {
            Field y_pos = new Field();
            Field y_neg = new Field();
            Link link = new Links.Utility.PosNeg(x, y_pos, y_neg);

            link.Forward();

            return (y_pos, y_neg);
        }
    }
}

namespace TensorShader.Links.Utility {
    /// <summary>正負</summary>
    public class PosNeg : Link{
        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>出力項1</summary>
        protected Field Y1 => OutFields[0];

        /// <summary>出力項2</summary>
        protected Field Y2 => OutFields[1];

        /// <summary>コンストラクタ</summary>
        public PosNeg(Field infield, Field outfield1, Field outfield2)
            : base(new Field[] { infield }, new Field[] { outfield1, outfield2 }) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y1.AssignValue(X.Value);
            Y2.AssignValue(-X.Value);
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y1.Grad == null || Y2.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Y1.Grad - Y2.Grad);
            }
        }

    }
}
