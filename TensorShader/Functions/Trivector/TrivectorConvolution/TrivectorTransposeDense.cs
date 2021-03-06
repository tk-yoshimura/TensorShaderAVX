using System;

namespace TensorShader {
    public abstract partial class VariableNode {
        /// <summary>3次元ベクトル転置全結合</summary>
        public static VariableNode TrivectorTransposeDense(VariableNode x, VariableNode w, bool gradmode = false) {
            Function function =
                new Functions.TrivectorConvolution.TrivectorTransposeDense(x.Shape, w.Shape, gradmode);

            VariableNode y = Apply(function, x, w)[0];

            return y;
        }
    }

    public partial class Tensor {
        /// <summary>3次元ベクトル転置全結合</summary>
        public static Tensor TrivectorTransposeDense(Tensor x, Tensor w, bool gradmode = false) {
            Functions.TrivectorConvolution.TrivectorTransposeDense function =
                new Functions.TrivectorConvolution.TrivectorTransposeDense(x.Shape, w.Shape, gradmode);

            Tensor y = new Tensor(function.OutShape);

            function.Execute(new Tensor[] { x, w }, new Tensor[] { y });

            return y;
        }
    }
}

namespace TensorShader.Functions.TrivectorConvolution {
    /// <summary>3次元ベクトル転置全結合</summary>
    internal class TrivectorTransposeDense : Function {
        /// <summary>入力特徴マップ形状</summary>
        public Shape InShape { private set; get; }

        /// <summary>出力特徴マップ形状</summary>
        public Shape OutShape { private set; get; }

        /// <summary>カーネル形状</summary>
        public Shape KernelShape { private set; get; }

        /// <summary>勾配</summary>
        public bool GradMode { private set; get; }

        /// <summary>コンストラクタ</summary>
        public TrivectorTransposeDense(Shape inshape, Shape kernelshape, bool gradmode) :
            base(inputs: 2, outputs: 1, allow_resubstitution : false) {
            if (inshape.Type != ShapeType.Map || inshape.Ndim != 2) {
                throw new ArgumentException(ExceptionMessage.TensorElements(inshape, ("Ndim", 2), ("Type", ShapeType.Map)));
            }

            if (kernelshape.Type != ShapeType.Kernel || kernelshape.Ndim != 2) {
                throw new ArgumentException(ExceptionMessage.TensorElements(kernelshape, ("Ndim", 2), ("Type", ShapeType.Kernel)));
            }

            if (inshape.Channels % 3 != 0) {
                throw new AggregateException(ExceptionMessage.TensorLengthMultiple("Channels", inshape, inshape.Channels, 3));
            }

            if (kernelshape.InChannels % 4 != 0) {
                throw new AggregateException(ExceptionMessage.TensorLengthMultiple("InChannels", kernelshape, kernelshape.Channels, 4));
            }

            if (inshape.Channels / 3 != kernelshape.OutChannels) {
                throw new ArgumentException(ExceptionMessage.TensorElements(kernelshape, ("InChannels", inshape.Channels / 3)));
            }

            this.InShape = inshape;
            this.OutShape = Shape.Map0D(kernelshape.InChannels / 4 * 3, inshape.Batch);
            this.KernelShape = kernelshape;
            this.GradMode = gradmode;
        }

        /// <summary>出力テンソル形状を返す</summary>
        public override Shape[] OutputShapes(params Shape[] inshapes) {
            CheckInputShapes(inshapes);

            return new Shape[] { OutShape };
        }

        public override void CheckInputShapes(params Shape[] inshapes) {
            base.CheckInputShapes(inshapes);

            if (inshapes[0] != InShape) {
                throw new ArgumentException(ExceptionMessage.ShapeWithIndex(index:0, inshapes[0], InShape));
            }

            if (inshapes[1] != KernelShape) {
                throw new ArgumentException(ExceptionMessage.ShapeWithIndex(index:1, inshapes[1], KernelShape));
            }
        }

        /// <summary>操作クラスを生成する</summary>
        internal override (Tensor[] tensors, Operator ope) GenerateOperator(Tensor[] intensors, Tensor[] outtensors) {
            CheckArgumentsCount(intensors, outtensors);

            return (new Tensor[] { intensors[0], intensors[1], outtensors[0] },
                    new Operators.TrivectorConvolution.TrivectorTransposeDense(
                        InShape.Channels, OutShape.Channels,
                        GradMode, InShape.Batch));
        }
    }
}
