using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShaderAvxBackend;

namespace TensorShaderAvxBackendTest.ArrayManipulations {
    [TestClass]
    public class CopyTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            for(uint copy_channels = 0; copy_channels <= 32; copy_channels++) {
                for(uint slides = 1; slides <= 100; slides++) {
                    float[] x = (new float[(copy_channels + 2) * slides]).Select((_) => (float)rd.NextDouble()).ToArray();
                    float[] y = (new float[(copy_channels + 4) * slides]).Select((_) => (float)rd.NextDouble()).ToArray();

                    float[] x_copy = (float[])x.Clone(), y_copy = (float[])y.Clone();

                    ArrayManipulation.PatternCopy(copy_channels + 2, 1, copy_channels + 4, 2, copy_channels, slides, x, y);

                    CollectionAssert.AreEqual(x_copy, x);

                    for(uint i = 0; i < slides; i++) {
                        Assert.AreEqual(y_copy[i * (copy_channels + 4) + 0], y[i * (copy_channels + 4) + 0], $"slides:{slides}, copys:{copy_channels}, index:{i}");
                        Assert.AreEqual(y_copy[i * (copy_channels + 4) + 1], y[i * (copy_channels + 4) + 1], $"slides:{slides}, copys:{copy_channels}, index:{i}");

                        for(uint j = 0; j < copy_channels; j++) {
                            Assert.AreEqual(x[i * (copy_channels + 2) + (j + 1)], y[i * (copy_channels + 4) + (j + 2)], $"slides:{slides}, copys:{copy_channels}, index:{i}, {j}");
                        }

                        Assert.AreEqual(y_copy[i * (copy_channels + 4) + (copy_channels + 2)], y[i * (copy_channels + 4) + (copy_channels + 2)], $"slides:{slides}, copys:{copy_channels}, index:{i}");
                        Assert.AreEqual(y_copy[i * (copy_channels + 4) + (copy_channels + 3)], y[i * (copy_channels + 4) + (copy_channels + 3)], $"slides:{slides}, copys:{copy_channels}, index:{i}");
                    }

                    Console.WriteLine($"pass:{slides}, {copy_channels}");
                }
            }
        }
    }
}
