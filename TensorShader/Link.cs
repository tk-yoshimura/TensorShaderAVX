using System;
using System.Collections.Generic;
using System.Linq;

namespace TensorShader {
    /// <summary>リンク</summary>
    public abstract class Link {
        private readonly List<Field> infields, outfields;

        /// <summary>入力フィールド</summary>
        public IReadOnlyList<Field> InFields => infields.AsReadOnly();

        /// <summary>入力フィールド数</summary>
        public int InFieldCount => infields.Count;

        /// <summary>出力フィールド</summary>
        public IReadOnlyList<Field> OutFields => outfields.AsReadOnly();

        /// <summary>出力フィールド数</summary>
        public int OutFieldCount => outfields.Count;

        /// <summary>リンク名</summary>
        public virtual string Name => GetType().Name.Split('.').Last();

        /// <summary>コンストラクタ</summary>
        public Link(Field[] infields, Field[] outfields) {
            foreach(Field infield in infields) {
                infield.InLinks.Add(this);
            }

            bool enable_backprob = infields.Select((field)=>field.EnableBackprop).Any((b)=>b);
            foreach(Field outfield in outfields) {
                if (outfield.OutLink != null) {
                    throw new ArgumentNullException(nameof(OutFields));
                }

                outfield.OutLink = this;
                outfield.EnableBackprop = enable_backprob;
            }

            this.infields = new List<Field>(infields);
            this.outfields = new List<Field>(outfields);
        }

        /// <summary>順伝搬</summary>
        public abstract void Forward();

        /// <summary>逆伝搬</summary>
        public virtual void Backward() {
            return;
        }

        /// <summary>文字列化</summary>
        public override string ToString() {
            return Name;
        }
    }
}
