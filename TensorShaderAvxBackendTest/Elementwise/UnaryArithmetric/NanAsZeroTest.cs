using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShaderAvxBackend;

namespace TensorShaderAvxBackendTest.ElementwiseLogicalArithmetric {
    [TestClass]
    public class NanAsZeroTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            for(int length = 0; length < 1000; length++) {
                float[] x = (new float[length + 2]).Select((_) => ((float)rd.Next(5) - 2) / ((float)rd.Next(5) - 2)).ToArray();
                float[] y = new float[length + 2];

                Elementwise.NanAsZero(1, (uint)length, x, y);

                for(int i = 1; i <= length; i++) {
                    Assert.AreEqual(float.IsNaN(x[i]) ? 0f : x[i], y[i], $"index:{i}");
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

            Elementwise.NanAsZero(0, (uint)x.Length, x, y);

            Assert.AreEqual(0, y[0]);
            Assert.AreEqual(1, y[1]);
            Assert.AreEqual(-1, y[2]);
            Assert.AreEqual(float.PositiveInfinity, y[3]);
            Assert.AreEqual(float.NegativeInfinity, y[4]);
            Assert.AreEqual(0, y[5]);
        }

        [TestMethod]
        public void SpeedTest() {
            int length = 10000000;

            float[] x = new float[length];
            float[] y = new float[length];

            Stopwatch sw = new Stopwatch();

            sw.Start();

            Elementwise.NanAsZero(0, (uint)length, x, y);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds} msec");
        }
    }
}
