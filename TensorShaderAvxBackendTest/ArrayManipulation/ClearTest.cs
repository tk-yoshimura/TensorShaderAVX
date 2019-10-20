using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShaderAvxBackend;

namespace TensorShaderAvxBackendTest.ArrayManipulations {
    [TestClass]
    public class ClearTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            for(int length = 0; length < 1000; length++) {
                float[] x = (new float[length + 2]).Select((_) => (float)rd.NextDouble() * 10 - 5).ToArray();

                ArrayManipulation.Clear(1, (uint)length, 0.5f, x);

                for(int i = 1; i <= length; i++) {
                    Assert.AreEqual(0.5f, x[i], $"index:{i}");
                }

                Assert.AreNotEqual(0.5f, x[0], $"index:0");
                Assert.AreNotEqual(0.5f, x[length + 1], $"index:{length + 1}");

                Console.WriteLine($"pass:{length}");
            }

        }

        [TestMethod]
        public void SpeedTest() {
            int length = 10000000;

            float[] x = new float[length];

            Stopwatch sw = new Stopwatch();

            sw.Start();

            ArrayManipulation.Clear(0, (uint)length, 0.5f, x);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds} msec");
        }
    }
}
