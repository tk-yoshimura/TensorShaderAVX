using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShaderAvxBackend;

namespace TensorShaderAvxBackendTest.ElementwiseUnaryArithmetric {
    [TestClass]
    public class SquareTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            for(int length = 0; length < 1000; length++) {
                float[] x = (new float[length + 1]).Select((_) => (float)rd.NextDouble() + 1).ToArray();
                float[] y = new float[length + 1];

                AvxArray<float> vx = x, vy = y;

                Elementwise.Square((uint)length, vx, vy);

                y = vy;

                for(int i = 0; i < length; i++) {
                    Assert.AreEqual(x[i] * x[i], y[i], 1e-4f, $"index:{i}");
                }

                Assert.AreEqual(0f, y[length], $"index:{length}");

                Console.WriteLine($"pass:{length}");
            }
        }

        [TestMethod]
        public void DenormalizedTest() {
            float[] x = new float[] { 0, 1, -1, float.PositiveInfinity, float.NegativeInfinity, float.NaN };
            float[] y = new float[x.Length];

            AvxArray<float> vx = x, vy = y;

            Elementwise.Square((uint)x.Length, vx, vy);

            y = vy;

            Assert.AreEqual(0, y[0]);
            Assert.AreEqual(1, y[1], 1e-4f);
            Assert.AreEqual(1, y[2], 1e-4f);
            Assert.AreEqual(float.PositiveInfinity, y[3]);
            Assert.AreEqual(float.PositiveInfinity, y[4]);
            Assert.IsTrue(float.IsNaN(y[5]));
        }

        [TestMethod]
        public void SpeedTest() {
            int length = 10000000;

            AvxArray<float> vx = new float[length];
            AvxArray<float> vy = new float[length];

            Stopwatch sw = new Stopwatch();

            sw.Start();

            Elementwise.Square((uint)length, vx, vy);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds} msec");
        }
    }
}
