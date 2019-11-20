using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShaderAvxBackend;

namespace TensorShaderAvxBackendTest.ElementwiseLogicalArithmetric {
    [TestClass]
    public class EqualTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            for(int length = 0; length < 1000; length++) {
                float[] x1 = (new float[length + 1]).Select((_) => (float)rd.Next(5) - 2).ToArray();
                float[] x2 = (new float[length + 1]).Select((_) => (float)rd.Next(5) - 2).ToArray();
                float[] y = new float[length + 1];

                AvxArray<float> vx1 = x1, vx2 = x2, vy = y;

                Elementwise.Equal((uint)length, vx1, vx2, vy);

                y = vy;

                for(int i = 0; i < length; i++) {
                    Assert.AreEqual(x1[i] == x2[i] ? 1f : 0f, y[i], $"index:{i}");
                }

                Assert.AreEqual(0f, y[length], $"index:{length}");

                Console.WriteLine($"pass:{length}");
            }
        }

        [TestMethod]
        public void SpeedTest() {
            int length = 10000000;

            AvxArray<float> vx1 = new float[length];
            AvxArray<float> vx2 = new float[length];
            AvxArray<float> vy = new float[length];

            Stopwatch sw = new Stopwatch();

            sw.Start();

            Elementwise.Equal((uint)length, vx1, vx2, vy);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds} msec");
        }
    }
}
