using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShaderAvxBackend;

namespace TensorShaderAvxBackendTest.ElementwiseBinaryArithmetric {
    [TestClass]
    public class ClampTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            for(int length = 0; length < 1000; length++) {
                float[] xval = (new float[length + 2]).Select((_) => (float)rd.NextDouble() * 2 - 1).ToArray();
                float[] xmin = (new float[length + 2]).Select((_) => (float)-rd.NextDouble()).ToArray();
                float[] xmax = (new float[length + 2]).Select((_) => (float)rd.NextDouble()).ToArray();
                float[] y = new float[length + 2];

                Elementwise.Clamp(1, (uint)length, xval, xmin, xmax, y);

                for(int i = 1; i <= length; i++) {
                    Assert.AreEqual(Math.Min(Math.Max(xval[i], xmin[i]), xmax[i]) , y[i], $"index:{i}");
                }

                Assert.AreEqual(0f, y[0], $"index:0");
                Assert.AreEqual(0f, y[length + 1], $"index:{length + 1}");

                Console.WriteLine($"pass:{length}");
            }
        }

        [TestMethod]
        public void SpeedTest() {
            int length = 10000000;

            float[] xval = new float[length];
            float[] xmin = new float[length];
            float[] xmax = new float[length];
            float[] y = new float[length];

            Stopwatch sw = new Stopwatch();

            sw.Start();

            Elementwise.Clamp(0, (uint)length, xval, xmin, xmax, y);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds} msec");
        }
    }
}
