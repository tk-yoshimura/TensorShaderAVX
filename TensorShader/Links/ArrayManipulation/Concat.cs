using System.Collections.Generic;
using System.Linq;
using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>結合</summary>
        public static Field Concat(int axis, params Field[] xs) {
            Field y = new Field();
            Link link = new Links.ArrayManipulation.Concat(axis, xs, y);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.ArrayManipulation {
    /// <summary>結合</summary>
    public class Concat : Link {
        /// <summary>入力項</summary>
        protected IReadOnlyList<Field> XS => InFields;

        /// <summary>出力項</summary>
        protected Field Y => OutFields[0];

        /// <summary>軸</summary>
        public int Axis { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Concat(int axis, Field[] infields, Field outfield)
            : base(infields, new Field[]{ outfield }) {
            this.Axis = axis;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Concat(Axis, XS.Select((field) => field.Value).ToArray()));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null || !XS.Select((field)=>field.EnableBackprop).Any()) {
                return;
            }

            VariableNode[] gxs = Separate(Y.Grad, Axis, XS.Select((field) => field.Shape[Axis]).ToArray());

            for (int i = 0; i < gxs.Length; i++) {
                if (XS[i].EnableBackprop) {
                    XS[i].AddGrad(gxs[i]);
                }
            }
        }
    }
}
