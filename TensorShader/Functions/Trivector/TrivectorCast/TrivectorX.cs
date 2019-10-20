using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>3次元ベクトルX成分</summary>
        public static VariableNode TrivectorX(VariableNode x) {
            Function function = new Functions.Trivector.TrivectorX();

            return Apply(function, x)[0];
        }
    }

    public partial class Tensor {
        /// <summary>3次元ベクトルX成分</summary>
        public static Tensor TrivectorX(Tensor x) {
            Function function = new Functions.Trivector.TrivectorX();

            Shape y_shape = function.OutputShapes(x.Shape)[0];

            Tensor y = new Tensor(y_shape);

            function.Execute(new Tensor[] { x }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.Trivector {
    /// <summary>3次元ベクトルX成分</summary>
    internal class TrivectorX : Function {
        /// <summary>コンストラクタ</summary>
        public TrivectorX()
            : base(inputs: 1, outputs: 1, allow_resubstitution : false){ }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            Shape inshape = inshapes[0];

            int[] s = inshape;
            s[0] /= 3;

            return new Shape[] { new Shape(inshape.Type, s) };
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);

            if (inshapes[0].Ndim <= 0) {
                throw new ArgumentException(ExceptionMessage.Shape("Ndim", inshapes[0]));
            }

            if (inshapes[0].Channels % 3 != 0) {
                throw new AggregateException(ExceptionMessage.TensorLengthMultiple("Channels", inshapes[0], inshapes[0].Channels, 3));
            }

            if (inshapes[0].InChannels % 3 != 0) {
                throw new AggregateException(ExceptionMessage.TensorLengthMultiple("InChannels", inshapes[0], inshapes[0].Channels, 3));
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            Tensor intensor = intensors[0], outtensor = outtensors[0];

            return (new Tensor[] { intensor, outtensor },
                    new Operators.TrivectorCast.TrivectorX(intensor.Shape));
        }
    }
}
