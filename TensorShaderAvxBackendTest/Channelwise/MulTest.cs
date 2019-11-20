using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShaderAvxBackend;

namespace TensorShaderAvxBackendTest.ChannelwiseBinaryArithmetric {
    [TestClass]
    public class MulTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            for(uint vector_length = 1; vector_length <= 100; vector_length++) {
                for(uint map_length = 0; map_length <= vector_length * 128; map_length += vector_length) {
                    float[] x1 = (new float[vector_length + 1]).Select((_) => (float)rd.NextDouble()).ToArray();
                    float[] x2 = (new float[map_length + 1]).Select((_) => (float)rd.NextDouble()).ToArray();
                    float[] y = new float[map_length + 1];
                    float v = y[map_length] = (float)rd.NextDouble();

                    AvxArray<float> vx1 = x1, vx2 = x2, vy = y;
                    
                    Channelwise.Mul(vector_length, map_length, vx1, vx2, vy);

                    x1 = vx1;  x2 = vx2;  y = vy;

                    for(uint i = 0; i < map_length; i++) {
                        Assert.AreEqual(x1[i % vector_length] * x2[i], y[i], 1e-5f, $"vector:{vector_length}, map:{map_length}, index:{i}");
                    }

                    Assert.AreEqual(v, y[map_length], $"vector:{vector_length}, map:{map_length}, last_index");

                    Console.WriteLine($"pass:{vector_length}, {map_length}");
                }
            }
        }
    }
}
