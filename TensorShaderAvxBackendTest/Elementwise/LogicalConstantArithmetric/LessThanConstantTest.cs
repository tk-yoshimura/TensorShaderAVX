using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShaderAvxBackend;

namespace TensorShaderAvxBackendTest.ElementwiseLogicalConstantArithmetric {
    [TestClass]
    public class LessThanConstantTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            for(int length = 0; length < 1000; length++) {
                float[] x = (new float[length + 2]).Select((_) => (float)rd.Next(5) - 2).ToArray();
                float[] y = new float[length + 2];

                Elementwise.LessThanConstant(1, (uint)length, -1, x, y);

                for(int i = 1; i <= length; i++) {
                    Assert.AreEqual(-1 < x[i] ? 1f : 0f, y[i], $"index:{i}");
                }

                Assert.AreEqual(0f, y[0], $"index:0");
                Assert.AreEqual(0f, y[length + 1], $"index:{length + 1}");

                Console.WriteLine($"pass:{length}");
            }
        }

        [TestMethod]
        public void SpeedTest() {
            int length = 10000000;

            float[] x = new float[length];
            float[] y = new float[length];

            Stopwatch sw = new Stopwatch();

            sw.Start();

            Elementwise.LessThanConstant(0, (uint)length, 0, x, y);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds} msec");
        }
    }
}
