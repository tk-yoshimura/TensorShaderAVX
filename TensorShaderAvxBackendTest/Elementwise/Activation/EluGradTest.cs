using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShaderAvxBackend;

namespace TensorShaderAvxBackendTest.ElementwiseActivation {
    [TestClass]
    public class EluGradTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            for(int length = 0; length < 1000; length++) {
                float[] x1 = (new float[length + 2]).Select((_) => (float)rd.NextDouble() * 2 - 1).ToArray();
                float[] x2 = (new float[length + 2]).Select((_) => (float)rd.NextDouble() * 2 - 1).ToArray();
                float[] y = new float[length + 2];

                Elementwise.EluGrad(1, (uint)length, 0.25f, x1, x2, y);

                for(int i = 1; i <= length; i++) {
                    Assert.AreEqual(x2[i] > 0 ? x1[i] : x1[i] * (0.25f * Math.Exp(x2[i])), y[i], 1e-4f, $"index:{i}");
                }

                Assert.AreEqual(0f, y[0], $"index:0");
                Assert.AreEqual(0f, y[length + 1], $"index:{length + 1}");

                Console.WriteLine($"pass:{length}");
            }
        }

        [TestMethod]
        public void DenormalizedTest() {
            float[] x1 = new float[] { 0, 1, -1, float.PositiveInfinity, float.NegativeInfinity, float.NaN };
            float[] x2 = new float[] { 1, 1, 1, 1, 1, 1 };
            float[] x3 = new float[] { -1, -1, -1, -1, -1, -1 };
            float[] y = new float[x1.Length];

            Elementwise.EluGrad(0, (uint)x1.Length, 0.25f, x1, x2, y);

            Assert.AreEqual(0, y[0]);
            Assert.AreEqual(1, y[1]);
            Assert.AreEqual(-1, y[2]);
            Assert.AreEqual(float.PositiveInfinity, y[3]);
            Assert.AreEqual(float.NegativeInfinity, y[4]);
            Assert.IsTrue(float.IsNaN(y[5]));

            Elementwise.EluGrad(0, (uint)x1.Length, 0.25f, x1, x3, y);

            Assert.AreEqual(0, y[0]);
            Assert.AreEqual(0.25f * Math.Exp(-1), y[1], 1e-4f);
            Assert.AreEqual(-0.25f * Math.Exp(-1), y[2], 1e-4f);
            Assert.AreEqual(float.PositiveInfinity, y[3]);
            Assert.AreEqual(float.NegativeInfinity, y[4]);
            Assert.IsTrue(float.IsNaN(y[5]));

            Elementwise.EluGrad(0, (uint)x1.Length, 0.25f, x2, x1, y);

            Assert.AreEqual(0, y[0]);
            Assert.AreEqual(1, y[1]);
            Assert.AreEqual(0.25f * Math.Exp(-1), y[2], 1e-4f);
            Assert.AreEqual(1, y[3]);
            Assert.AreEqual(0, y[4]);
            Assert.IsTrue(float.IsNaN(y[5]));

            Elementwise.EluGrad(0, (uint)x1.Length, 0.25f, x3, x1, y);

            Assert.AreEqual(0, y[0]);
            Assert.AreEqual(-1, y[1]);
            Assert.AreEqual(-0.25f * Math.Exp(-1), y[2], 1e-4f);
            Assert.AreEqual(-1, y[3]);
            Assert.AreEqual(0, y[4]);
            Assert.IsTrue(float.IsNaN(y[5]));
        }

        [TestMethod]
        public void SpeedTest() {
            int length = 10000000;

            float[] x1 = new float[length];
            float[] x2 = new float[length];
            float[] y = new float[length];

            Stopwatch sw = new Stopwatch();

            sw.Start();

            Elementwise.EluGrad(0, (uint)length, 0.25f, x1, x2, y);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds} msec");
        }
    }
}
