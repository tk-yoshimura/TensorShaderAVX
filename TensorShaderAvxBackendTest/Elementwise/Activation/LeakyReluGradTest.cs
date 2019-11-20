using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShaderAvxBackend;

namespace TensorShaderAvxBackendTest.ElementwiseActivation {
    [TestClass]
    public class LeakyReluGradTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            for(int length = 0; length < 1000; length++) {
                float[] x1 = (new float[length + 1]).Select((_) => (float)rd.NextDouble() * 2 - 1).ToArray();
                float[] x2 = (new float[length + 1]).Select((_) => (float)rd.NextDouble() * 2 - 1).ToArray();
                float[] y = new float[length + 1];

                AvxArray<float> vx1 = x1, vx2 = x2, vy = y;

                Elementwise.LeakyReluGrad((uint)length, 0.5f, vx1, vx2, vy);

                x1 = vx1; x2 = vx2; y = vy;

                for(int i = 0; i < length; i++) {
                    Assert.AreEqual(x2[i] > 0 ? x1[i] : x1[i] * 0.5f, y[i], $"index:{i}");
                }

                Assert.AreEqual(0f, y[length], $"index:{length}");

                Console.WriteLine($"pass:{length}");
            }
        }

        [TestMethod]
        public void DenormalizedTest() {
            AvxArray<float> vx1 = new float[] { 0, 1, -1, float.PositiveInfinity, float.NegativeInfinity, float.NaN };
            AvxArray<float> vx2 = new float[] { 1, 1, 1, 1, 1, 1 };
            AvxArray<float> vx3 = new float[] { -1, -1, -1, -1, -1, -1 };
            float[] y;
            AvxArray<float> vy = new AvxArray<float>(vx1.Length);

            Elementwise.LeakyReluGrad((uint)vx1.Length, 0.25f, vx1, vx2, vy);

            y = vy;

            Assert.AreEqual(0, y[0]);
            Assert.AreEqual(1, y[1]);
            Assert.AreEqual(-1, y[2]);
            Assert.AreEqual(float.PositiveInfinity, y[3]);
            Assert.AreEqual(float.NegativeInfinity, y[4]);
            Assert.IsTrue(float.IsNaN(y[5]));

            Elementwise.LeakyReluGrad((uint)vx1.Length, 0.25f, vx1, vx3, vy);

            y = vy;

            Assert.AreEqual(0, y[0]);
            Assert.AreEqual(0.25f, y[1], 1e-4f);
            Assert.AreEqual(-0.25f, y[2], 1e-4f);
            Assert.AreEqual(float.PositiveInfinity, y[3]);
            Assert.AreEqual(float.NegativeInfinity, y[4]);
            Assert.IsTrue(float.IsNaN(y[5]));

            Elementwise.LeakyReluGrad((uint)vx1.Length, 0.25f, vx2, vx1, vy);

            y = vy;

            Assert.AreEqual(0, y[0]);
            Assert.AreEqual(1, y[1]);
            Assert.AreEqual(0.25f, y[2], 1e-4f);
            Assert.AreEqual(1, y[3]);
            Assert.AreEqual(0.25f, y[4]);
            Assert.AreEqual(0.25f, y[5]);

            Elementwise.LeakyReluGrad((uint)vx1.Length, 0.25f, vx3, vx1, vy);

            y = vy;

            Assert.AreEqual(0, y[0]);
            Assert.AreEqual(-1, y[1]);
            Assert.AreEqual(-0.25f, y[2], 1e-4f);
            Assert.AreEqual(-1, y[3]);
            Assert.AreEqual(-0.25f, y[4]);
            Assert.AreEqual(-0.25f, y[5]);
        }

        [TestMethod]
        public void SpeedTest() {
            int length = 10000000;

            AvxArray<float> vx1 = new float[length];
            AvxArray<float> vx2 = new float[length];
            AvxArray<float> vy = new float[length];

            Stopwatch sw = new Stopwatch();

            sw.Start();

            Elementwise.LeakyReluGrad((uint)length, 0.5f, vx1, vx2, vy);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds} msec");
        }
    }
}
