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
                float[] x = (new float[length + 1]).Select((_) => (float)rd.NextDouble() * 2 - 1).ToArray();
                float[] y = new float[length + 1];

                AvxArray<float> vx = x, vy = y;

                Elementwise.ClampConstant((uint)length, -0.5f, 0.5f, vx, vy);

                y = vy;

                for(int i = 0; i < length; i++) {
                    Assert.AreEqual(Math.Min(Math.Max(x[i], -0.5f), 0.5f), y[i], $"index:{i}");
                }

                Assert.AreEqual(0f, y[length], $"index:{length}");

                Console.WriteLine($"pass:{length}");
            }
        }

        [TestMethod]
        public void SpeedTest() {
            int length = 10000000;

            AvxArray<float> vx = new float[length];
            AvxArray<float> vy = new float[length];

            Stopwatch sw = new Stopwatch();

            sw.Start();

            Elementwise.ClampConstant((uint)length, -0.5f, 0.5f, vx, vy);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds} msec");
        }
    }
}
