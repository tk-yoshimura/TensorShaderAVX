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
                float[] x = (new float[length + 1]).Select((_) => (float)rd.NextDouble() * 10 - 5).ToArray();

                AvxArray<float> vx = x;

                ArrayManipulation.Clear((uint)length, 0.5f, vx);

                x = vx;

                for(int i = 0; i < length; i++) {
                    Assert.AreEqual(0.5f, x[i], $"index:{i}");
                }

                Assert.AreNotEqual(0.5f, x[length], $"index:{length}");

                Console.WriteLine($"pass:{length}");
            }

        }

        [TestMethod]
        public void SpeedTest() {
            int length = 10000000;

            AvxArray<float> vx = new float[length];

            Stopwatch sw = new Stopwatch();

            sw.Start();

            ArrayManipulation.Clear((uint)length, 0.5f, vx);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds} msec");
        }
    }
}
