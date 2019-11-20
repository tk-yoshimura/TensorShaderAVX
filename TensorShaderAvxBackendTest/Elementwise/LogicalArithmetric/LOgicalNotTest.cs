using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShaderAvxBackend;

namespace TensorShaderAvxBackendTest.ElementwiseUnaryArithmetric {
    [TestClass]
    public class LogicalNotTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            for(int length = 0; length < 1000; length++) {
                float[] x = (new float[length + 1]).Select((_) => (float)rd.NextDouble()).ToArray();
                float[] y = new float[length + 1];

                AvxArray<float> vx = x, vy = y;

                Elementwise.LogicalNot((uint)length, vx, vy);

                y = vy;

                for(int i = 0; i < length; i++) {
                    Assert.AreEqual(1 - x[i], y[i], 1e-4f, $"index:{i}");
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

            Elementwise.LogicalNot((uint)length, vx, vy);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds} msec");
        }
    }
}
