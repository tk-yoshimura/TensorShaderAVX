using System;
using System.Linq;

namespace TensorShader {
    /// <summary>テンソルクラス</summary>
    public partial class Tensor : ICloneable{
        /// <summary>形状</summary>
        public Shape Shape{ protected set; get; }

        /// <summary>保有バッファ</summary>
        internal float[] Buffer { private protected set; get; }

        /// <summary>形状タイプ</summary>
        public ShapeType Type => Shape.Type;

        /// <summary>要素数</summary>
        public int Length => Shape.Length;

        /// <summary>次元数</summary>
        public int Ndim => Shape.Ndim;

        /// <summary>幅</summary>
        public int Width => Shape.Width;

        /// <summary>高さ</summary>
        public int Height => Shape.Height;

        /// <summary>奥行き</summary>
        public int Depth => Shape.Depth;

        /// <summary>チャネル数</summary>
        public int Channels => Shape.Channels;

        /// <summary>入力チャネル数</summary>
        public int InChannels => Shape.InChannels;

        /// <summary>出力チャネル数</summary>
        public int OutChannels => Shape.OutChannels;

        /// <summary>バッチ数</summary>
        public int Batch => Shape.Batch;

        /// <summary>コンストラクタ</summary>
        /// <param name="shape">形状</param>
        /// <param name="value">初期値(任意指定)</param>
        /// <param name="clone_value">初期値配列をディープコピーするか</param>
        protected Tensor(Shape shape, float[] value, bool clone_value){
            if (shape == null) {
                throw new ArgumentException(nameof(shape));
            }
            if (value != null && value.Length < shape.Length) {
                throw new ArgumentException(ExceptionMessage.Argument("value.Length", value.Length, shape.Length));
            }
            if (value == null && clone_value) {
                throw new ArgumentException(nameof(clone_value));
            }

            this.Buffer = (value == null)
                            ? new float[shape.Length]
                            : (clone_value ? value : (float[])value.Clone());
            this.Shape = shape;
        }

        /// <summary>コンストラクタ</summary>
        /// <param name="shape">形状</param>
        /// <param name="value">初期値(任意指定)</param>
        public Tensor(Shape shape, float[] value = null)
            : this(shape, value, clone_value: false) { }

        /// <summary>状態</summary>
        public virtual float[] State{
            get {
                float[] state = new float[Length];
                Array.Copy(Buffer, state, Length);

                return state;
            }
            set {
                if (value.Length != Length) {
                    throw new ArgumentException(ExceptionMessage.Argument("value.Length", value.Length, Length));
                }

                Array.Copy(value, Buffer, Length);
            }
        }

        /// <summary>初期化</summary>
        public virtual void Clear(float val) {
            TensorShaderAvxBackend.ArrayManipulation.Clear(0, (uint)Length, val, Buffer);
        }

        /// <summary>初期化</summary>
        public void Zeroset() {
            TensorShaderAvxBackend.ArrayManipulation.Zeroset(0, (uint)Length, Buffer);
        }

        /// <summary>コピー</summary>
        /// <remarks>形状の一致不一致を問わず実行される</remarks>
        public virtual void CopyTo(Tensor tensor) {
            if (tensor.Length < Length) {
                throw new ArgumentException(ExceptionMessage.Argument("tensor.Length", tensor.Length, Length));
            }
            if (ReferenceEquals(Buffer, tensor.Buffer)) {
                return;
            }

            Array.Copy(Buffer, tensor.Buffer, Length);
        }

        /// <summary>部分コピー</summary>
        /// <remarks>
        /// 形状の一致不一致を問わず実行される
        /// コピー先に同じテンソルを指定時、領域が重なる場合の動作は不定
        /// </remarks>
        public void RegionCopyTo(Tensor tensor, uint src_index, uint dst_index, uint count) {
            if (src_index >= Length || src_index + count > Length) {
                throw new ArgumentOutOfRangeException();
            }
            if (dst_index >= tensor.Length || dst_index + count > tensor.Length) {
                throw new ArgumentOutOfRangeException();
            }

            Array.Copy(Buffer, src_index, tensor.Buffer, dst_index, count);
        }

        /// <summary>形状変更</summary>
        public void Reshape(Shape shape) {
            if (Length != shape.Length) {
                throw new ArgumentException(nameof(shape));
            }

            this.Shape = shape;
        }

        /// <summary>クローン</summary>
        public object Clone() {
            return Copy();
        }

        /// <summary>コピー</summary>
        public virtual Tensor Copy() {
            if(this is OverflowCheckedTensor) {
                OverflowCheckedTensor tensor = new OverflowCheckedTensor(Shape);
                CopyTo(tensor);

                return tensor;
            }
            else {
                Tensor tensor = new Tensor(Shape);
                CopyTo(tensor);

                return tensor;
            }
        }

        /// <summary>インデクサ</summary>
        /// <param name="index">バッチ次元のインデクス</param>
        public virtual Tensor this[int index]{
            set {
                if (index < 0 || index >= Batch) {
                    throw new ArgumentOutOfRangeException(nameof(index));
                }

                if ((Type, Width, Height, Depth, Channels, InChannels, OutChannels)
                    != (value.Type, value.Width, value.Height, value.Depth, value.Channels, value.InChannels, value.OutChannels)) {
                    throw new ArgumentException(nameof(value));
                }

                value.RegionCopyTo(this, 0, (uint)(Shape.DataSize * index), (uint)Shape.DataSize);
            }

            get {
                if (index < 0 || index >= Batch) {
                    throw new ArgumentOutOfRangeException(nameof(index));
                }

                int[] s = Shape;

                if (s.Length >= 2 && Batch > 1) {
                    s[s.Length - 1] = 1;

                    Tensor tensor = (this is OverflowCheckedTensor)
                                    ? new OverflowCheckedTensor(new Shape(Type, s))
                                    : new Tensor(new Shape(Type, s));

                    RegionCopyTo(tensor, (uint)(Shape.DataSize * index), 0, (uint)Shape.DataSize);

                    return tensor;
                }
                else {
                    return Copy();
                }
            }
        }

        /// <summary>定数</summary>
        public static Tensor Constant(Shape shape, float val) {
            Tensor tensor = new Tensor(shape);
            tensor.Clear(val);

            return tensor;
        }
    }

    /// <summary>バッファ共有テンソル</summary>
    internal class TemporaryTensor : Tensor {
        public TemporaryTensor(Shape shape, float[] value = null) : 
            base(shape, value, clone_value:true){ }
    }

    /// <summary>領域外アクセスチェック有効テンソル</summary>
    /// <remarks>デバック用</remarks>
    internal class OverflowCheckedTensor : Tensor{
        private static Random random = new Random();

        private const int canary_length = 32;
        private readonly float[] canary;

        /// <summary>コンストラクタ</summary>
        /// <param name="shape">形状</param>
        /// <param name="value">初期値(任意指定)</param>
        public OverflowCheckedTensor(Shape shape, float[] value = null)
            : base(shape, new float[shape.Length + canary_length]){
            if (shape == null) {
                throw new ArgumentException(nameof(shape));
            }
            if(value != null) {
                if(value.Length != shape.Length) {
                    throw new ArgumentException(ExceptionMessage.Argument("value.Length", value.Length, shape.Length));
                }
                Array.Copy(value, Buffer, shape.Length);
            }
            else {
                Buffer[0] = (float)random.NextDouble();
            }

            this.canary = (new float[canary_length]).Select((_) => (float)random.NextDouble()).ToArray();

            Array.Copy(canary, 0, Buffer, shape.Length, canary_length);

            this.Shape = shape;
        }

        /// <summary>状態</summary>
        /// <exception cref="IndexOutOfRangeException">領域外アクセスが検知されたとき</exception>
        public override float[] State {
            get {
                CheckOverflow();

                float[] value = new float[Length];
                Array.Copy(Buffer, value, Length);

                return value;
            }

            set {
                if (value.Length != Length) {
                    throw new ArgumentException(ExceptionMessage.Argument("value.Length", value.Length, Length));
                }

                Array.Copy(value, Buffer, Length);
            }
        }

        /// <summary>コピー</summary>
        public override void CopyTo(Tensor tensor) {
            Array.Copy(Buffer, tensor.Buffer, (uint)Length);
        }

        /// <summary>領域外アクセスチェック</summary>
        public void CheckOverflow() {
            for(int i = 0; i < canary_length; i++) {
                if (Buffer[Length + i] != canary[i]) {
                    throw new IndexOutOfRangeException("Detected out of buffer access.");
                }
            }
        }
    }
}
