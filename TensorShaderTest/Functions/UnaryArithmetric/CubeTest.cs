using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;

namespace TensorShaderTest.Functions.UnaryArithmetric {
    [TestClass]
    public class CubeTest {
        [TestMethod]
        public void ExecuteTest() {
            const int length = 256;
            Random rd = new Random(1234);

            int[] idxes = (new int[length]).Select((_, idx) => idx).ToArray();

            float[] x = (new float[length]).Select((_) => (float)rd.NextDouble() * 2f - 1f).ToArray();

            {
                Tensor t = new Tensor(Shape.Vector(length), x);

                Tensor o = Tensor.Cube(t);

                AssertError.Tolerance(idxes.Select((idx) => x[idx] * x[idx] * x[idx]).ToArray(), o.State, 1e-7f, 1e-5f);
            }

            {
                InputNode t = new Tensor(Shape.Vector(length), x);

                var n = t + 0;

                OutputNode o = VariableNode.Cube(n).Save();

                Flow flow = Flow.FromOutputs(o);
                flow.Execute();

                AssertError.Tolerance(idxes.Select((idx) => x[idx] * x[idx] * x[idx]).ToArray(), o.State, 1e-7f, 1e-5f);
            }
        }
    }
}
