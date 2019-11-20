using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShaderAvxBackend;

namespace TensorShaderAvxBackendTest.ArrayManipulations {
    [TestClass]
    public class BroadcastTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            for (uint src_length = 1; src_length <= 32; src_length++) {
                for (uint dst_length = src_length; dst_length <= src_length * 32; dst_length += src_length) {
                    for (uint slides = 0; slides <= 4; slides++) {
                        float[] x = (new float[src_length * slides + 1]).Select((_) => (float)rd.NextDouble() * 2 - 1).ToArray();
                        float[] y = (new float[dst_length * slides + 1]).Select((_) => (float)rd.NextDouble() + 1).ToArray();

                        float v1 = y[dst_length * slides];

                        AvxArray<float> vx = x, vy = y;

                        ArrayManipulation.Broadcast(src_length, vx, dst_length, vy, slides);

                        y = vy;

                        for (int j = 0; j < slides; j++) {
                            for (int i = 0; i < dst_length; i++) {
                                Assert.AreEqual(x[i % src_length + j * src_length], y[i + j * dst_length]);
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

            for (uint src_length = 1; src_length <= 32; src_length++) {
                uint dst_length = src_length * 1000000;

                AvxArray<float> vx = new float[src_length];
                AvxArray<float> vy = new float[dst_length];

                ArrayManipulation.Broadcast(src_length, vx, dst_length, vy, 1);

                sw.Restart();

                ArrayManipulation.Broadcast(src_length, vx, dst_length, vy, 1);
                ArrayManipulation.Broadcast(src_length, vx, dst_length, vy, 1);
                ArrayManipulation.Broadcast(src_length, vx, dst_length, vy, 1);
                ArrayManipulation.Broadcast(src_length, vx, dst_length, vy, 1);

                sw.Stop();

                Console.WriteLine($"{dst_length} : {sw.ElapsedMilliseconds / 4} msec");
            }
        }
    }
}
