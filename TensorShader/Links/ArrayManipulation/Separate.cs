using System.Collections.Generic;
using System.Linq;
using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>分離</summary>
        public static IEnumerable<Field> Separate(Field x, int axis, params int[] lengths) {
            Field[] ys = (new Field[lengths.Length]).Select((_) => new Field()).ToArray();
            Link link = new Links.ArrayManipulation.Separate(axis, x, ys, lengths);

            link.Forward();

            return ys;
        }
    }
}

namespace TensorShader.Links.ArrayManipulation {
    /// <summary>分離</summary>
    public class Separate : Link {
        private readonly int[] lengths;

        /// <summary>入力項</summary>
        protected Field X => InFields[0];

        /// <summary>出力項</summary>
        protected IReadOnlyList<Field> YS => OutFields;

        /// <summary>軸</summary>
        public int Axis { private set; get; }

        /// <summary>コンストラクタ</summary>
        public Separate(int axis, Field infield, Field[] outfields, params int[] lengths)
            : base(new Field[] { infield }, outfields) {
            this.lengths = lengths;
            this.Axis = axis;
        }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            VariableNode[] ys = Separate(X.Value, Axis, lengths);

            for (int i = 0; i < ys.Length; i++) {
                YS[i].AssignValue(ys[i]);
            }
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            VariableNode[] gys = YS.Select((field) => field.Grad).ToArray();

            if (gys.Any((grad) => grad == null)) {
                return;
            }

            if (X.EnableBackprop) {
                X.AddGrad(Concat(Axis, gys));
            }
        }
    }
}
