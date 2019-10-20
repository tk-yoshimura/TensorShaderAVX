using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShaderAvxBackend;

namespace TensorShaderAvxBackendTest.ElementwiseMultiArithmetric {
    [TestClass]
    public class SumTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            for(int length = 0; length < 1000; length++) {
                for (int n = 0; n < 12; n++) {
                    float[][] xs = new float[n][];

                    for (int j = 0; j < n; j++) {
                        xs[j] = (new float[length + 2]).Select((_) => (float)rd.NextDouble()).ToArray();
                    }

                    float[] y = new float[length + 2];

                    Elementwise.Sum(1, (uint)length, xs, y);

                    for (int i = 1; i <= length; i++) {
                        float sum = 0;
                        for (int j = 0; j < n; j++) {
                            sum += xs[j][i];
                        }

                        Assert.AreEqual(sum, y[i], 1e-5f, $"index:{i}");
                    }

                    Assert.AreEqual(0f, y[0], $"index:0");
                    Assert.AreEqual(0f, y[length + 1], $"index:{length + 1}");

                    Console.WriteLine($"pass:{length}, {n}");
                }
            }
        }

        [TestMethod]
        public void SpeedTest() {
            int length = 10000000;

            for (int n = 0; n < 12; n++) {
                float[][] xs = (new float[n][]).Select((_) => new float[length]).ToArray();
                float[] y = new float[length];

                Stopwatch sw = new Stopwatch();

                sw.Start();

                Elementwise.Sum(0, (uint)length, xs, y);

                sw.Stop();

                Console.WriteLine($"{n} : {sw.ElapsedMilliseconds} msec");
            }
        }
    }
}
