using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShaderAvxBackend;

namespace TensorShaderAvxBackendTest.ElementwiseBinaryArithmetric {
    [TestClass]
    public class DivTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            for(int length = 0; length < 1000; length++) {
                float[] x1 = (new float[length + 2]).Select((_) => (float)rd.NextDouble()).ToArray();
                float[] x2 = (new float[length + 2]).Select((_) => (float)rd.NextDouble() + 1f).ToArray();
                float[] y = new float[length + 2];

                Elementwise.Div(1, (uint)length, x1, x2, y);

                for(int i = 1; i <= length; i++) {
                    Assert.AreEqual(x1[i] / x2[i], y[i], $"index:{i}");
                }

                Assert.AreEqual(0f, y[0], $"index:0");
                Assert.AreEqual(0f, y[length + 1], $"index:{length + 1}");

                Console.WriteLine($"pass:{length}");
            }
        }

        [TestMethod]
        public void SpeedTest() {
            {
                int length = 10000000;

                float[] x1 = new float[length];
                float[] x2 = new float[length];
                float[] y = new float[length];

                Stopwatch sw = new Stopwatch();

                sw.Start();

                Elementwise.Div(0, (uint)length, x1, x2, y);

                sw.Stop();

                Console.WriteLine($"{sw.ElapsedMilliseconds} msec");
            }
        }
    }
}
