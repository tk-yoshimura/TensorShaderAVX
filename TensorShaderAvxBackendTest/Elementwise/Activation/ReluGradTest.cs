using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShaderAvxBackend;

namespace TensorShaderAvxBackendTest.ElementwiseActivation {
    [TestClass]
    public class ReluGradTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            for(int length = 0; length < 1000; length++) {
                float[] x1 = (new float[length + 1]).Select((_) => (float)rd.NextDouble() * 2 - 1).ToArray();
                float[] x2 = (new float[length + 1]).Select((_) => (float)rd.NextDouble() * 2 - 1).ToArray();
                float[] y = new float[length + 1];

                AvxArray<float> vx1 = x1, vx2 = x2, vy = y;

                Elementwise.ReluGrad((uint)length, vx1, vx2, vy);

                y = vy;

                for(int i = 0; i < length; i++) {
                    Assert.AreEqual(x2[i] > 0 ? x1[i] : 0f, y[i], $"index:{i}");
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

            Elementwise.ReluGrad((uint)vx1.Length, vx1, vx2, vy);

            y = vy;

            Assert.AreEqual(0, y[0]);
            Assert.AreEqual(1, y[1]);
            Assert.AreEqual(-1, y[2]);
            Assert.AreEqual(float.PositiveInfinity, y[3]);
            Assert.AreEqual(float.NegativeInfinity, y[4]);
            Assert.IsTrue(float.IsNaN(y[5]));

            Elementwise.ReluGrad((uint)vx1.Length, vx1, vx3, vy);

            y = vy;

            Assert.AreEqual(0, y[0]);
            Assert.AreEqual(0, y[1]);
            Assert.AreEqual(0, y[2]);
            Assert.AreEqual(0, y[3]);
            Assert.AreEqual(0, y[4]);
            Assert.AreEqual(0, y[5]);

            Elementwise.ReluGrad((uint)vx1.Length, vx2, vx1, vy);

            y = vy;

            Assert.AreEqual(0, y[0]);
            Assert.AreEqual(1, y[1]);
            Assert.AreEqual(0, y[2], 1e-4f);
            Assert.AreEqual(1, y[3]);
            Assert.AreEqual(0, y[4]);
            Assert.AreEqual(0, y[5]);

            Elementwise.ReluGrad((uint)vx1.Length, vx3, vx1, vy);

            y = vy;

            Assert.AreEqual(0, y[0]);
            Assert.AreEqual(-1, y[1]);
            Assert.AreEqual(0, y[2]);
            Assert.AreEqual(-1, y[3]);
            Assert.AreEqual(0, y[4]);
            Assert.AreEqual(0, y[5]);
        }

        [TestMethod]
        public void SpeedTest() {
            int length = 10000000;

            AvxArray<float> vx1 = new float[length];
            AvxArray<float> vx2 = new float[length];
            AvxArray<float> vy = new float[length];

            Stopwatch sw = new Stopwatch();

            sw.Start();

            Elementwise.ReluGrad((uint)length, vx1, vx2, vy);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds} msec");
        }
    }
}
