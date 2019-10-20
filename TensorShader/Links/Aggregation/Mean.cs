using static TensorShader.VariableNode;

namespace TensorShader {
    public partial class Field {
        /// <summary>平均</summary>
        public static Field Mean(Field x, int[] axes = null, bool keepdims = false) {
            Field y = new Field();
            Link link = new Links.Aggregation.Mean(x, y, axes, keepdims);

            link.Forward();

            return y;
        }
    }
}

namespace TensorShader.Links.Aggregation {
    /// <summary>平均</summary>
    public class Mean : Aggregation {
        /// <summary>コンストラクタ</summary>
        public Mean(Field infield, Field outfield, int[] axes = null, bool keepdims = false)
            : base(infield, outfield, axes, keepdims) { }

        /// <summary>順伝搬</summary>
        public override void Forward() {
            Y.AssignValue(Mean(X.Value, Axes, KeepDims));
        }

        /// <summary>逆伝搬</summary>
        public override void Backward() {
            if (Y.Grad == null) {
                return;
            }

            if (X.EnableBackprop) {
                VariableNode g = AdjustShape(Y.Grad, X.Shape);

                X.AddGrad(g / (X.Shape.Length / Y.Shape.Length));
            }
        }
    }
}
