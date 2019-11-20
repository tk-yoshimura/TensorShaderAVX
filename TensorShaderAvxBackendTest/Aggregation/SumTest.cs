using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShaderAvxBackend;

namespace TensorShaderAvxBackendTest.Aggregations {
    [TestClass]
    public class SumTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            for (uint dst_length = 1; dst_length <= 32; dst_length++) {
                for (uint src_length = dst_length; src_length <= dst_length * 32; src_length += dst_length) {
                    for (uint slides = 0; slides <= 4; slides++) {
                        float[] x = (new float[src_length * slides + 1]).Select((_) => (float)rd.NextDouble() * 2 - 1).ToArray();
                        float[] y = (new float[dst_length * slides + 1]).Select((_) => (float)rd.NextDouble() + 1).ToArray();

                        float v1 = y[dst_length * slides];

                        AvxArray<float> vx = x, vy = y;

                        Aggregation.Sum(src_length, vx, dst_length, vy, slides);

                        y = vy;

                        for (int j = 0; j < slides; j++) {
                            double[] sum = new double[dst_length];

                            for (int i = 0; i < src_length; i++) {
                                sum[i % dst_length] += x[i + j * src_length];
                            }

                            for (int i = 0; i < dst_length; i++) {
                                Assert.AreEqual(sum[i], y[i + j * dst_length], 1e-5f);
                            }
                        }

                        Assert.AreEqual(v1, y[dst_length * slides]);

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

                AvxArray<float> vx = new float[src_length];
                AvxArray<float> vy = new float[dst_length];

                Aggregation.Sum(src_length, vx, dst_length, vy, 1);

                sw.Restart();

                Aggregation.Sum(src_length, vx, dst_length, vy, 1);
                Aggregation.Sum(src_length, vx, dst_length, vy, 1);
                Aggregation.Sum(src_length, vx, dst_length, vy, 1);
                Aggregation.Sum(src_length, vx, dst_length, vy, 1);

                sw.Stop();

                Console.WriteLine($"{dst_length} : {sw.ElapsedMilliseconds / 4} msec");
            }
        }
    }
}
