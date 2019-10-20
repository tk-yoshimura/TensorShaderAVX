using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShaderAvxBackend;

namespace TensorShaderAvxBackendTest.Aggregations {
    [TestClass]
    public class AverageTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            for (uint dst_length = 1; dst_length <= 32; dst_length++) {
                for (uint src_length = dst_length; src_length <= dst_length * 32; src_length += dst_length) {
                    for (uint slides = 0; slides <= 4; slides++) {
                        float[] x = (new float[src_length * slides + 2]).Select((_) => (float)rd.NextDouble() * 2 - 1).ToArray();
                        float[] y = (new float[dst_length * slides + 2]).Select((_) => (float)rd.NextDouble() + 1).ToArray();

                        float v0 = y[0], v1 = y[dst_length * slides + 1];

                        Aggregation.Average(1, src_length, x, 1, dst_length, y, slides);

                        double inv = (double)dst_length / src_length;

                        for (int j = 0; j < slides; j++) {
                            double[] sum = new double[dst_length];

                            for (int i = 0; i < src_length; i++) {
                                sum[i % dst_length] += x[i + j * src_length + 1];
                            }

                            for (int i = 0; i < dst_length; i++) {
                                Assert.AreEqual(sum[i] * inv, y[i + j * dst_length + 1], 1e-5f);
                            }
                        }

                        Assert.AreEqual(v0, y[0]);
                        Assert.AreEqual(v1, y[dst_length * slides + 1]);

                        Console.WriteLine($"pass: {src_length} {dst_length} {slides}");
                    }
                }
            }
        }

        [TestMethod]
        public void SpeedTest() {
            Random rd = new Random(1234);

            Stopwatch sw = new Stopwatch();

            for (uint dst_length = 1; dst_length <= 32; dst_length++) {
                uint src_length = dst_length * 1000000;

                float[] x = new float[src_length];
                float[] y = new float[dst_length];

                Aggregation.Average(0, src_length, x, 0, dst_length, y, 1);

                sw.Restart();

                Aggregation.Average(0, src_length, x, 0, dst_length, y, 1);
                Aggregation.Average(0, src_length, x, 0, dst_length, y, 1);
                Aggregation.Average(0, src_length, x, 0, dst_length, y, 1);
                Aggregation.Average(0, src_length, x, 0, dst_length, y, 1);

                sw.Stop();

                Console.WriteLine($"{dst_length} : {sw.ElapsedMilliseconds / 4} msec");
            }
        }
    }
}
