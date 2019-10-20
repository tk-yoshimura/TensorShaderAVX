using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShaderAvxBackend;

namespace TensorShaderAvxBackendTest.ElementwiseBinaryConstantArithmetric {
    [TestClass]
    public class ClampConstantTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            for(int length = 0; length < 1000; length++) {
                float[] x = (new float[length + 2]).Select((_) => (float)rd.NextDouble() * 2 - 1).ToArray();
                float[] y = new float[length + 2];

                Elementwise.ClampConstant(1, (uint)length, -0.5f, 0.5f, x, y);

                for(int i = 1; i <= length; i++) {
                    Assert.AreEqual(Math.Min(Math.Max(x[i], -0.5f), 0.5f), y[i], $"index:{i}");
                }

                Assert.AreEqual(0f, y[0], $"index:0");
                Assert.AreEqual(0f, y[length + 1], $"index:{length + 1}");

                Console.WriteLine($"pass:{length}");
            }
        }

        [TestMethod]
        public void SpeedTest() {
            int length = 10000000;

            float[] x = new float[length];
            float[] y = new float[length];

            Stopwatch sw = new Stopwatch();

            sw.Start();

            Elementwise.ClampConstant(0, (uint)length, -0.5f, 0.5f, x, y);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds} msec");
        }
    }
}
