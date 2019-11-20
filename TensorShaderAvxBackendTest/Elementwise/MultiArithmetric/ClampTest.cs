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
                float[] xval = (new float[length + 1]).Select((_) => (float)rd.NextDouble() * 2 - 1).ToArray();
                float[] xmin = (new float[length + 1]).Select((_) => (float)-rd.NextDouble()).ToArray();
                float[] xmax = (new float[length + 1]).Select((_) => (float)rd.NextDouble()).ToArray();
                float[] y = new float[length + 1];

                AvxArray<float> vxval = xval, vxmin = xmin, vxmax = xmax, vy = y;

                Elementwise.Clamp((uint)length, vxval, vxmin, vxmax, vy);

                y = vy;

                for(int i = 0; i < length; i++) {
                    Assert.AreEqual(Math.Min(Math.Max(xval[i], xmin[i]), xmax[i]) , y[i], $"index:{i}");
                }

                Assert.AreEqual(0f, y[length], $"index:{length}");

                Console.WriteLine($"pass:{length}");
            }
        }

        [TestMethod]
        public void SpeedTest() {
            int length = 10000000;

            AvxArray<float> vxval = new float[length];
            AvxArray<float> vxmin = new float[length];
            AvxArray<float> vxmax = new float[length];
            AvxArray<float> vy = new float[length];

            Stopwatch sw = new Stopwatch();

            sw.Start();

            Elementwise.Clamp((uint)length, vxval, vxmin, vxmax, vy);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds} msec");
        }
    }
}
