using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;

namespace TensorShaderTest.Functions.UnaryArithmetric {
    [TestClass]
    public class SignedPowTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 256;
            Random rd = new Random(1234);

            int[] idxes = (new int[length]).Select((_, idx) => idx).ToArray();

            float[] x = (new float[length]).Select((_) => (float)rd.NextDouble() * 2f - 1f).ToArray();

            {
                Tensor t = new Tensor(Shape.Vector(length), x);

                Tensor o = Tensor.SignedPow(t, 1.5f);

                AssertError.Tolerance(idxes.Select((idx) => Math.Sign(x[idx]) * (float)Math.Pow(Math.Abs(x[idx]), 1.5f)).ToArray(), o.State, 1e-7f, 1e-5f);
            }

            {
                InputNode t = new Tensor(Shape.Vector(length), x);

                var n = t + 0;

                OutputNode o = VariableNode.SignedPow(n, 1.5f).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => Math.Sign(x[idx]) * (float)Math.Pow(Math.Abs(x[idx]), 1.5f)).ToArray(), o.State, 1e-7f, 1e-5f);
            }
        }
    }
}
