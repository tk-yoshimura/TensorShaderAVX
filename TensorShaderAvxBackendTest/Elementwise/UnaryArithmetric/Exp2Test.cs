using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShaderAvxBackend;

namespace TensorShaderAvxBackendTest.ElementwiseUnaryArithmetric {
    [TestClass]
    public class Exp2Test {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            for(int length = 0; length < 1000; length++) {
                float[] x = (new float[length + 2]).Select((_) => (float)rd.NextDouble() * 10 - 5).ToArray();
                float[] y = new float[length + 2];

                Elementwise.Exp2(1, (uint)length, x, y);

                for(int i = 1; i <= length; i++) {
                    Assert.AreEqual(Math.Pow(2, x[i]), y[i], 1e-3f, $"index:{i}");
                }

                Assert.AreEqual(0f, y[0], $"index:0");
                Assert.AreEqual(0f, y[length + 1], $"index:{length + 1}");

                Console.WriteLine($"pass:{length}");
            }
        }

        [TestMethod]
        public void DenormalizedTest() {
            float[] x = new float[] { 0, 1, -1, float.PositiveInfinity, float.NegativeInfinity, float.NaN };
            float[] y = new float[x.Length];

            Elementwise.Exp2(0, (uint)x.Length, x, y);

            Assert.AreEqual(1, y[0]);
            Assert.AreEqual(2, y[1], 1e-4f);
            Assert.AreEqual(0.5f, y[2], 1e-4f);
            Assert.AreEqual(float.PositiveInfinity, y[3]);
            Assert.AreEqual(0, y[4]);
            Assert.IsTrue(float.IsNaN(y[5]));
        }

        [TestMethod]
        public void SpeedTest() {
            int length = 10000000;

            float[] x = new float[length];
            float[] y = new float[length];

            Stopwatch sw = new Stopwatch();

            sw.Start();

            Elementwise.Exp2(0, (uint)length, x, y);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds} msec");
        }
    }
}
