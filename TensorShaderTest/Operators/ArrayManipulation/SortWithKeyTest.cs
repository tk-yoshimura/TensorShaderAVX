using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.ArrayManipulation;

namespace TensorShaderTest.Operators.ArrayManipulation {
    [TestClass]
    public class SortWithKeyTest {
        [TestMethod]
        public void ExecuteTest() {
            Random rd = new Random(1234);

            {
                Shape shape = new Shape(ShapeType.Map, 17, 8, 2, 4, 1, 3, 5, 67);

                for (int axis = 0; axis < shape.Ndim; axis++) {
                    int stride = 1, axislength = shape[axis];
                    for(int i = 0; i < axis; i++) {
                        stride *= shape[i];
                    }

                    float[] x1 = (new float[shape.Length]).Select((_, idx) => (float)(((idx * 4969 % 17 + 3) * (idx * 6577 % 13 + 5) + idx) % 8)).ToArray();
                    float[] x2 = (new float[shape.Length]).Select((_, idx) => (float)(idx / stride % axislength)).ToArray();

                    OverflowCheckedTensor inkey  = new OverflowCheckedTensor(shape, x1);
                    OverflowCheckedTensor inval  = new OverflowCheckedTensor(shape, x2);
                    OverflowCheckedTensor outkey = new OverflowCheckedTensor(shape);
                    OverflowCheckedTensor outval = new OverflowCheckedTensor(shape);

                    SortWithKey ope = new SortWithKey(shape, axis);

                    ope.Execute(inkey, inval, outkey, outval);

                    CollectionAssert.AreEqual(x1, inkey.State);
                    CollectionAssert.AreEqual(x2, inval.State);

                    float[] y = outkey.State;
                    int p = 0;

                    for (int i = 0; i < shape.Length / axislength; i++, p = i / stride * stride * axislength + i % stride) {
                        for (int j = 1; j < axislength; j++) {
                            if (y[p + (j - 1) * stride] > y[p + j * stride]) {
                                Assert.Fail($"axis:{axis} outkey");
                            }
                        }
                    }

                    Assert.AreEqual(shape.Length, p);

                    ope.Execute(outval, outkey, inval, inkey);

                    float[] y2 = inval.State;
                    p = 0;

                    for (int i = 0; i < shape.Length / axislength; i++, p = i / stride * stride * axislength + i % stride) {
                        for (int j = 1; j < axislength; j++) {
                            if (y2[p + (j - 1) * stride] > y2[p + j * stride]) {
                                Assert.Fail($"axis:{axis} inval");
                            }
                        }
                    }

                    Assert.AreEqual(shape.Length, p);

                    float[] y3 = inkey.State;

                    CollectionAssert.AreEqual(x1, y3, $"axis:{axis} inkey");

                    Console.WriteLine($"pass : axis{axis}");
                }
            }
        }

        [TestMethod]
        public void SpeedTest() {
            Shape shape = new Shape(ShapeType.Map, 8195, 500);
            int length = shape.Length, axis = 0, stride = 1, axislength = shape[axis];

            float[] xval = (new float[length]).Select((_, idx) => (float)(((idx * 4969 % 17 + 3) * (idx * 6577 % 13 + 5) + idx) % 8)).ToArray();
            float[] ival = (new float[shape.Length]).Select((_, idx) => (float)(idx / stride % shape[axis])).ToArray();

            OverflowCheckedTensor x_key = new OverflowCheckedTensor(shape, xval);
            OverflowCheckedTensor x_val = new OverflowCheckedTensor(shape, ival);

            OverflowCheckedTensor y_key = new OverflowCheckedTensor(shape);
            OverflowCheckedTensor y_val = new OverflowCheckedTensor(shape);

            SortWithKey ope = new SortWithKey(shape, axis);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            ope.Execute(x_key, x_val, y_key, y_val);

            sw.Stop();

            float[] v = y_key.State;

            for(int i = 1; i < axislength; i++) {
                Assert.IsTrue(v[i - 1] <= v[i], $"{i}: {v[i - 1]}, {v[i]}");
                Assert.IsTrue(v[i - 1 + axislength] <= v[i + axislength], $"{i + axislength}: {v[i - 1 + axislength]}, {v[i + axislength]}");
                Assert.IsTrue(v[i - 1 + axislength * 2] <= v[i + axislength * 2], $"{i + axislength * 2}: {v[i - 1 + axislength * 2]}, {v[i + axislength * 2]}");
                Assert.IsTrue(v[i - 1 + axislength * 3] <= v[i + axislength * 3], $"{i + axislength * 3}: {v[i - 1 + axislength * 3]}, {v[i + axislength * 3]}");
                Assert.IsTrue(v[i - 1 + axislength * 4] <= v[i + axislength * 4], $"{i + axislength * 4}: {v[i - 1 + axislength * 4]}, {v[i + axislength * 4]}");
            }

            Console.WriteLine($"{sw.ElapsedMilliseconds} msec");
        }

        [TestMethod]
        public void ReverseTest() {
            Shape shape = new Shape(ShapeType.Map, 8195, 500);
            int length = shape.Length, axis = 0, stride = 1, axislength = shape[axis];

            float[] xval = (new float[length]).Select((_, idx) => (float)(idx)).Reverse().ToArray();
            float[] ival = (new float[shape.Length]).Select((_, idx) => (float)(idx / stride % shape[axis])).ToArray();

            OverflowCheckedTensor x_key = new OverflowCheckedTensor(shape, xval);
            OverflowCheckedTensor x_val = new OverflowCheckedTensor(shape, ival);

            OverflowCheckedTensor y_key = new OverflowCheckedTensor(shape);
            OverflowCheckedTensor y_val = new OverflowCheckedTensor(shape);

            SortWithKey ope = new SortWithKey(shape, axis);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            ope.Execute(x_key, x_val, y_key, y_val);

            sw.Stop();

            float[] v = y_key.State;

            for(int i = 1; i < axislength; i++) {
                Assert.IsTrue(v[i - 1] <= v[i], $"{i}: {v[i - 1]}, {v[i]}");
                Assert.IsTrue(v[i - 1 + axislength] <= v[i + axislength], $"{i + axislength}: {v[i - 1 + axislength]}, {v[i + axislength]}");
                Assert.IsTrue(v[i - 1 + axislength * 2] <= v[i + axislength * 2], $"{i + axislength * 2}: {v[i - 1 + axislength * 2]}, {v[i + axislength * 2]}");
                Assert.IsTrue(v[i - 1 + axislength * 3] <= v[i + axislength * 3], $"{i + axislength * 3}: {v[i - 1 + axislength * 3]}, {v[i + axislength * 3]}");
                Assert.IsTrue(v[i - 1 + axislength * 4] <= v[i + axislength * 4], $"{i + axislength * 4}: {v[i - 1 + axislength * 4]}, {v[i + axislength * 4]}");
            }

            Console.WriteLine($"{sw.ElapsedMilliseconds} msec");
        }
    }
}
