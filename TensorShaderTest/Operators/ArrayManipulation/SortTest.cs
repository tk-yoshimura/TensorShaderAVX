using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.ArrayManipulation;

namespace TensorShaderTest.Operators.ArrayManipulation {
    [TestClass]
    public class SortTest {
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

                    OverflowCheckedTensor inval  = new OverflowCheckedTensor(shape, x1);
                    OverflowCheckedTensor outval = new OverflowCheckedTensor(shape);

                    Sort ope = new Sort(shape, axis);

                    ope.Execute(inval, outval);

                    CollectionAssert.AreEqual(x1, inval.State);

                    float[] y = outval.State;
                    int p = 0;

                    for (int i = 0; i < shape.Length / axislength; i++, p = i / stride * stride * axislength + i % stride) {
                        for (int j = 1; j < axislength; j++) {
                            if (y[p + (j - 1) * stride] > y[p + j * stride]) {
                                Assert.Fail($"axis:{axis} outval");
                            }
                        }
                    }

                    Assert.AreEqual(shape.Length, p);

                    Console.WriteLine($"pass : axis{axis}");
                }
            }
        }

        [TestMethod]
        public void ReferenceTest() {
            Shape shape = new Shape(ShapeType.Map, 3, 4, 2, 7, 8, 5);
            int length = shape.Length;

            float[] xval = (new float[length]).Select((_, idx) => (float)(((idx * 4969 % 17 + 3) * (idx * 6577 % 13 + 5) + idx) % 8)).ToArray();
            OverflowCheckedTensor x = new OverflowCheckedTensor(shape, xval);
            OverflowCheckedTensor y = new OverflowCheckedTensor(shape);

            Sort ope = new Sort(shape, axis:3);

            ope.Execute(x, y);

            float[] y_actual = y.State;

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"not equal");
        }

        [TestMethod]
        public void SpeedTest() {
            Shape shape = Shape.Map0D(8192, 500);
            int length = shape.Length, axislength = shape[0];

            float[] xval = (new float[length]).Select((_, idx) => (float)(((idx * 4969 % 17 + 3) * (idx * 6577 % 13 + 5) + idx) % 8)).ToArray();
            OverflowCheckedTensor x = new OverflowCheckedTensor(shape, xval);
            OverflowCheckedTensor y = new OverflowCheckedTensor(shape);

            Sort ope = new Sort(shape, axis:0);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            ope.Execute(x, y);

            sw.Stop();

            float[] v = y.State;

            for(int i = 1; i < axislength / 5; i++) {
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
            Shape shape = Shape.Map0D(8192, 500);
            int length = shape.Length, axislength = shape[0];

            float[] xval = (new float[length]).Select((_, idx) => (float)idx).Reverse().ToArray();
            OverflowCheckedTensor x = new OverflowCheckedTensor(shape, xval);
            OverflowCheckedTensor y = new OverflowCheckedTensor(shape);

            Sort ope = new Sort(shape, axis:0);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            ope.Execute(x, y);

            sw.Stop();

            float[] v = y.State;

            for(int i = 1; i < axislength / 5; i++) {
                Assert.IsTrue(v[i - 1] <= v[i], $"{i}: {v[i - 1]}, {v[i]}");
                Assert.IsTrue(v[i - 1 + axislength] <= v[i + axislength], $"{i + axislength}: {v[i - 1 + axislength]}, {v[i + axislength]}");
                Assert.IsTrue(v[i - 1 + axislength * 2] <= v[i + axislength * 2], $"{i + axislength * 2}: {v[i - 1 + axislength * 2]}, {v[i + axislength * 2]}");
                Assert.IsTrue(v[i - 1 + axislength * 3] <= v[i + axislength * 3], $"{i + axislength * 3}: {v[i - 1 + axislength * 3]}, {v[i + axislength * 3]}");
                Assert.IsTrue(v[i - 1 + axislength * 4] <= v[i + axislength * 4], $"{i + axislength * 4}: {v[i - 1 + axislength * 4]}, {v[i + axislength * 4]}");
            }

            Console.WriteLine($"{sw.ElapsedMilliseconds} msec");
        }

        float[] y_expect = {
            0,  1,  0,  1,  0,  1,  2,  0,  0,  0,  0,  0,  0,  0,  2,  1,  0,  0,  1,  2,  0,  1,  0,  1,  1,  1,  2,  3,  0,  1,  3,  1,  0,  1,  0,  1,  0,  0,  2,  2,  0,  1,  2,  3,  1,  1,  1,  1,
            2,  1,  2,  3,  3,  1,  4,  3,  1,  1,  2,  3,  4,  1,  2,  3,  0,  1,  2,  3,  2,  1,  2,  2,  3,  1,  3,  5,  4,  2,  5,  3,  2,  3,  2,  4,  4,  2,  3,  4,  1,  2,  2,  4,  2,  2,  5,  3,
            4,  7,  4,  5,  4,  4,  5,  6,  2,  3,  2,  5,  4,  3,  6,  6,  4,  3,  4,  5,  2,  5,  6,  7,  5,  7,  5,  5,  5,  5,  6,  7,  2,  6,  2,  6,  4,  3,  6,  7,  6,  3,  4,  7,  3,  5,  6,  7,
            7,  7,  7,  6,  6,  5,  6,  7,  7,  7,  5,  7,  6,  6,  6,  7,  6,  6,  4,  7,  4,  7,  7,  7,  0,  0,  2,  0,  2,  0,  4,  0,  0,  0,  0,  0,  0,  5,  0,  0,  0,  1,  0,  0,  0,  3,  0,  1,
            0,  1,  2,  1,  4,  1,  4,  2,  0,  1,  1,  1,  1,  5,  1,  3,  0,  1,  2,  2,  1,  3,  0,  1,  0,  2,  2,  3,  4,  1,  4,  3,  2,  1,  2,  1,  3,  5,  2,  5,  1,  1,  3,  3,  2,  3,  2,  4,
            0,  4,  2,  4,  6,  2,  6,  5,  2,  1,  3,  2,  4,  6,  3,  5,  2,  4,  4,  3,  3,  4,  2,  5,  4,  5,  5,  5,  6,  4,  6,  7,  3,  3,  6,  3,  5,  7,  4,  7,  3,  5,  5,  7,  4,  5,  5,  6,
            4,  6,  6,  6,  6,  5,  6,  7,  5,  5,  6,  3,  6,  7,  5,  7,  5,  5,  6,  7,  4,  5,  6,  7,  4,  7,  6,  7,  7,  7,  7,  7,  6,  5,  7,  3,  7,  7,  6,  7,  6,  7,  7,  7,  6,  6,  7,  7,
            1,  0,  0,  2,  0,  1,  1,  1,  0,  1,  1,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  1,  1,  2,  1,  1,  3,  1,  1,  4,  2,  0,  2,  2,  1,  0,  2,  2,  2,  0,  2,  2,  1,  2,  1,  2,  1,
            4,  1,  2,  3,  2,  1,  4,  3,  2,  3,  2,  2,  2,  3,  2,  3,  0,  3,  2,  3,  4,  3,  3,  2,  4,  1,  3,  3,  3,  5,  5,  3,  3,  3,  6,  3,  4,  4,  6,  4,  4,  3,  4,  3,  4,  5,  4,  4,
            5,  7,  4,  5,  4,  5,  6,  4,  4,  4,  6,  3,  4,  5,  6,  5,  6,  4,  4,  6,  4,  5,  6,  5,  7,  7,  6,  7,  4,  5,  6,  7,  5,  5,  6,  4,  4,  7,  6,  6,  6,  5,  4,  7,  5,  5,  6,  7,
            7,  7,  7,  7,  7,  6,  7,  7,  6,  7,  7,  7,  5,  7,  6,  7,  6,  6,  5,  7,  7,  6,  6,  7,  0,  1,  0,  0,  4,  1,  2,  1,  0,  1,  2,  1,  1,  2,  0,  3,  0,  0,  0,  2,  2,  1,  0,  0,
            0,  2,  0,  1,  4,  4,  3,  3,  0,  1,  2,  1,  4,  3,  0,  3,  1,  1,  1,  3,  2,  2,  1,  1,  0,  3,  0,  2,  4,  4,  4,  4,  1,  3,  2,  3,  4,  5,  1,  3,  4,  1,  2,  3,  3,  3,  2,  1,
            3,  4,  2,  3,  6,  5,  4,  5,  2,  4,  3,  3,  4,  5,  2,  6,  5,  1,  2,  5,  4,  3,  2,  2,  4,  5,  2,  4,  6,  5,  4,  6,  4,  5,  4,  3,  6,  5,  4,  7,  6,  1,  3,  5,  4,  5,  3,  3,
            4,  5,  2,  5,  6,  6,  6,  7,  4,  6,  5,  4,  7,  7,  5,  7,  7,  5,  4,  7,  5,  5,  6,  5,  4,  6,  6,  6,  6,  7,  6,  7,  7,  7,  6,  6,  7,  7,  7,  7,  7,  5,  7,  7,  7,  7,  6,  7,
            0,  1,  0,  3,  0,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  1,  2,  2,  0,  1,  0,  0,  0,  1,  2,  3,  1,  1,  1,  3,  0,  1,  2,  1,  0,  1,  0,  1,  0,  2,  2,  3,  0,  2,  2,  1,
            2,  1,  3,  3,  2,  1,  2,  5,  1,  1,  2,  1,  0,  4,  1,  2,  0,  3,  2,  3,  0,  3,  2,  1,  2,  1,  4,  3,  2,  2,  3,  5,  2,  1,  3,  1,  1,  5,  2,  3,  3,  4,  4,  4,  0,  3,  5,  6,
            5,  4,  5,  5,  5,  5,  5,  7,  2,  3,  4,  2,  4,  5,  6,  3,  6,  4,  4,  5,  1,  4,  6,  7,  6,  7,  6,  7,  7,  5,  6,  7,  5,  3,  6,  3,  4,  6,  6,  4,  6,  6,  4,  6,  4,  4,  6,  7,
            7,  7,  7,  7,  7,  5,  6,  7,  7,  3,  6,  7,  6,  7,  6,  7,  6,  7,  5,  7,  4,  5,  7,  7,  0,  0,  0,  0,  4,  0,  2,  0,  0,  1,  1,  0,  2,  1,  0,  3,  0,  1,  0,  1,  0,  3,  0,  1,
            0,  1,  0,  2,  4,  1,  4,  1,  0,  2,  2,  1,  2,  1,  0,  3,  1,  1,  1,  3,  1,  3,  1,  1,  0,  2,  1,  3,  4,  2,  4,  2,  0,  3,  2,  2,  2,  4,  3,  6,  4,  1,  3,  3,  3,  3,  4,  2,
            2,  5,  2,  4,  5,  3,  6,  4,  3,  4,  6,  3,  3,  5,  4,  7,  5,  1,  4,  3,  4,  5,  6,  4,  4,  5,  2,  5,  6,  4,  6,  5,  4,  5,  6,  3,  4,  6,  5,  7,  6,  3,  5,  5,  4,  5,  6,  5,
            4,  6,  2,  6,  6,  5,  6,  7,  4,  6,  6,  5,  4,  7,  6,  7,  6,  5,  6,  5,  6,  5,  6,  7,  7,  7,  7,  7,  6,  7,  7,  7,  6,  7,  7,  7,  5,  7,  6,  7,  7,  5,  7,  5,  7,  6,  7,  7,
            0,  0,  0,  1,  1,  1,  1,  3,  0,  1,  0,  1,  0,  0,  0,  0,  0,  0,  2,  0,  0,  0,  1,  0,  1,  1,  0,  3,  2,  1,  2,  5,  1,  1,  2,  3,  2,  1,  0,  1,  0,  1,  2,  1,  0,  2,  2,  1,
            2,  1,  1,  3,  3,  5,  3,  5,  2,  3,  2,  3,  4,  2,  1,  3,  0,  2,  2,  2,  0,  3,  2,  2,  2,  5,  2,  3,  4,  5,  4,  5,  2,  3,  5,  3,  4,  3,  3,  3,  4,  5,  3,  3,  4,  4,  3,  3,
            3,  5,  2,  4,  6,  5,  5,  7,  3,  3,  6,  4,  4,  5,  4,  4,  6,  6,  4,  5,  4,  5,  6,  4,  4,  5,  3,  7,  6,  5,  6,  7,  4,  4,  6,  6,  5,  5,  6,  6,  6,  7,  6,  6,  4,  5,  6,  5,
            6,  6,  6,  7,  7,  7,  7,  7,  5,  7,  7,  7,  7,  5,  6,  7,  7,  7,  6,  7,  5,  6,  6,  7,  0,  1,  0,  0,  1,  0,  2,  0,  0,  0,  1,  1,  0,  0,  0,  2,  0,  1,  0,  0,  0,  1,  0,  1,
            0,  1,  0,  3,  2,  1,  2,  3,  0,  1,  2,  1,  1,  1,  0,  3,  0,  1,  1,  3,  1,  3,  2,  1,  2,  1,  2,  3,  4,  3,  2,  4,  0,  1,  2,  2,  2,  1,  1,  3,  1,  1,  2,  3,  3,  3,  3,  3,
            3,  2,  2,  3,  4,  4,  5,  5,  0,  2,  4,  3,  2,  2,  2,  3,  2,  2,  2,  3,  3,  3,  4,  3,  4,  3,  2,  5,  4,  6,  6,  6,  4,  3,  4,  3,  4,  5,  4,  5,  3,  5,  4,  5,  4,  5,  5,  6,
            5,  4,  3,  6,  6,  7,  6,  6,  4,  6,  6,  4,  4,  5,  6,  7,  4,  7,  5,  5,  5,  5,  6,  7,  6,  5,  5,  6,  6,  7,  6,  7,  7,  7,  6,  6,  6,  7,  7,  7,  5,  7,  7,  5,  6,  5,  7,  7,
            0,  0,  0,  0,  0,  0,  0,  2,  0,  1,  1,  0,  0,  1,  0,  1,  0,  1,  2,  1,  0,  0,  2,  0,  0,  1,  0,  3,  3,  1,  1,  5,  1,  1,  2,  3,  1,  1,  0,  1,  0,  1,  2,  1,  0,  1,  4,  1,
            0,  1,  0,  3,  4,  5,  2,  5,  2,  1,  2,  3,  2,  4,  5,  4,  0,  3,  2,  4,  0,  2,  4,  4,  2,  2,  1,  5,  4,  5,  2,  5,  3,  3,  3,  7,  2,  5,  6,  6,  1,  4,  4,  5,  0,  4,  6,  5,
            2,  3,  2,  7,  5,  5,  3,  7,  3,  3,  4,  7,  3,  5,  6,  7,  3,  5,  6,  6,  4,  5,  6,  6,  2,  5,  6,  7,  6,  7,  6,  7,  5,  3,  5,  7,  3,  6,  6,  7,  6,  6,  6,  6,  4,  6,  6,  7,
            7,  5,  7,  7,  7,  7,  7,  7,  6,  4,  6,  7,  4,  7,  7,  7,  6,  7,  7,  7,  4,  7,  6,  7,  0,  0,  0,  1,  0,  2,  2,  0,  0,  0,  0,  0,  2,  0,  0,  0,  0,  0,  1,  3,  1,  0,  0,  2,
            1,  1,  1,  1,  0,  3,  2,  3,  0,  2,  2,  2,  2,  3,  0,  1,  0,  1,  2,  3,  2,  1,  1,  3,  2,  1,  2,  1,  3,  4,  5,  4,  0,  3,  2,  3,  2,  5,  1,  3,  4,  1,  2,  3,  3,  3,  2,  3,
            3,  5,  2,  2,  4,  5,  6,  5,  0,  4,  2,  3,  4,  5,  3,  3,  4,  5,  3,  4,  4,  3,  3,  7,  4,  5,  4,  3,  5,  5,  6,  5,  2,  5,  4,  5,  4,  5,  4,  6,  5,  6,  4,  5,  6,  5,  4,  7,
            5,  5,  6,  3,  6,  7,  6,  6,  4,  6,  4,  6,  4,  6,  6,  7,  6,  7,  5,  5,  6,  5,  6,  7,  6,  6,  7,  4,  6,  7,  6,  7,  4,  7,  4,  7,  5,  7,  6,  7,  7,  7,  6,  5,  7,  5,  7,  7,
            0,  1,  0,  1,  0,  0,  0,  3,  0,  1,  0,  3,  1,  0,  0,  1,  0,  0,  0,  0,  0,  1,  4,  0,  0,  1,  2,  3,  1,  2,  2,  5,  1,  1,  1,  3,  2,  1,  1,  1,  2,  1,  2,  1,  0,  1,  4,  1,
            2,  4,  2,  3,  2,  3,  2,  5,  4,  1,  2,  3,  3,  1,  2,  2,  4,  1,  2,  2,  3,  2,  4,  1,  2,  5,  2,  3,  4,  5,  3,  6,  5,  2,  4,  4,  4,  2,  3,  3,  4,  2,  2,  3,  4,  3,  5,  2,
            2,  5,  3,  4,  4,  5,  5,  7,  6,  3,  5,  7,  4,  5,  4,  5,  4,  7,  3,  3,  4,  4,  6,  3,  3,  6,  5,  6,  4,  7,  6,  7,  6,  5,  6,  7,  5,  5,  6,  7,  5,  7,  6,  5,  4,  5,  6,  4,
            6,  7,  6,  7,  7,  7,  7,  7,  7,  5,  7,  7,  7,  5,  7,  7,  7,  7,  6,  6,  4,  6,  6,  6,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  2,  0,  0,  1,  0,  1,  0,  1,  2,  0,  0,  1,  2,  3,
            0,  1,  0,  1,  0,  1,  2,  1,  0,  1,  2,  2,  2,  2,  0,  2,  0,  1,  2,  1,  2,  1,  3,  3,  1,  1,  1,  1,  1,  3,  2,  2,  0,  1,  2,  2,  2,  3,  2,  3,  0,  1,  4,  3,  4,  1,  4,  5,
            2,  3,  2,  3,  4,  5,  2,  3,  1,  3,  4,  3,  2,  4,  2,  4,  1,  2,  5,  3,  5,  4,  4,  7,  5,  3,  2,  3,  4,  6,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  3,  4,  5,  5,  6,  5,  5,  7,
            6,  5,  3,  5,  6,  7,  6,  7,  6,  6,  4,  5,  4,  6,  6,  7,  4,  7,  6,  5,  6,  5,  6,  7,  7,  5,  5,  7,  7,  7,  6,  7,  6,  7,  7,  7,  4,  7,  6,  7,  4,  7,  7,  6,  7,  5,  7,  7,
            0,  0,  2,  0,  0,  4,  0,  0,  0,  1,  0,  3,  0,  1,  0,  1,  0,  0,  1,  2,  0,  0,  1,  0,  0,  1,  2,  1,  0,  5,  0,  2,  0,  1,  0,  3,  1,  3,  3,  1,  0,  1,  2,  3,  2,  1,  4,  1,
            0,  1,  6,  1,  3,  5,  3,  5,  2,  1,  3,  3,  3,  3,  4,  1,  1,  1,  2,  3,  4,  2,  4,  2,  2,  2,  6,  2,  4,  5,  5,  5,  3,  3,  4,  7,  4,  5,  5,  1,  2,  5,  4,  4,  4,  5,  6,  2,
            2,  3,  6,  2,  4,  6,  6,  7,  4,  5,  5,  7,  5,  5,  6,  6,  4,  6,  6,  5,  6,  5,  6,  4,  2,  4,  6,  3,  5,  7,  6,  7,  5,  5,  5,  7,  6,  5,  6,  7,  7,  7,  6,  7,  6,  6,  6,  7,
            3,  5,  7,  7,  6,  7,  6,  7,  6,  6,  6,  7,  7,  5,  7,  7,  7,  7,  7,  7,  7,  7,  6,  7,  1,  1,  1,  1,  2,  0,  0,  3,  0,  0,  2,  0,  0,  0,  1,  0,  0,  0,  0,  2,  1,  1,  2,  1,
            2,  1,  2,  1,  4,  2,  2,  3,  0,  1,  2,  1,  2,  1,  2,  2,  0,  1,  0,  3,  3,  1,  3,  3,  3,  1,  2,  1,  4,  3,  2,  4,  4,  1,  2,  2,  2,  2,  2,  2,  4,  3,  0,  3,  4,  4,  4,  3,
            4,  3,  4,  3,  4,  5,  5,  5,  5,  2,  3,  3,  4,  3,  6,  3,  4,  4,  2,  4,  4,  5,  4,  4,  5,  3,  5,  3,  5,  5,  6,  6,  6,  3,  4,  5,  4,  5,  6,  4,  4,  5,  2,  5,  6,  5,  5,  7,
            6,  3,  6,  3,  6,  7,  6,  7,  6,  4,  4,  5,  4,  6,  6,  5,  4,  6,  3,  5,  6,  5,  6,  7,  7,  7,  7,  4,  7,  7,  7,  7,  7,  5,  4,  6,  7,  7,  6,  7,  5,  7,  6,  7,  6,  7,  7,  7,
            0,  0,  2,  0,  0,  0,  0,  1,  0,  1,  0,  2,  0,  3,  0,  1,  0,  1,  0,  0,  1,  1,  2,  0,  0,  1,  2,  1,  0,  1,  0,  3,  0,  1,  1,  3,  0,  3,  0,  1,  0,  1,  2,  1,  2,  1,  3,  3,
            0,  3,  2,  2,  1,  2,  1,  3,  1,  1,  2,  3,  1,  3,  1,  1,  3,  1,  2,  3,  4,  2,  4,  5,  1,  4,  3,  3,  4,  3,  2,  3,  6,  2,  2,  3,  2,  4,  2,  5,  4,  2,  2,  3,  4,  4,  4,  5,
            2,  5,  6,  3,  4,  5,  4,  4,  6,  5,  4,  3,  3,  5,  3,  7,  5,  5,  3,  6,  5,  5,  5,  6,  4,  6,  6,  4,  4,  6,  6,  6,  6,  5,  5,  7,  4,  5,  5,  7,  6,  7,  4,  7,  6,  5,  6,  7,
            4,  7,  6,  6,  7,  7,  6,  7,  7,  7,  7,  7,  5,  5,  7,  7,  7,  7,  5,  7,  7,  6,  6,  7,  0,  1,  1,  1,  0,  1,  0,  0,  0,  1,  0,  1,  0,  1,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  1,  1,  1,  1,  1,  1,  2,  0,  1,  2,  3,  0,  2,  2,  1,  0,  1,  0,  3,  2,  1,  1,  3,  2,  1,  2,  1,  2,  4,  2,  3,  0,  3,  2,  4,  0,  3,  4,  2,  0,  1,  0,  3,  4,  1,  2,  5,
            3,  3,  3,  3,  3,  5,  3,  3,  1,  4,  4,  5,  3,  3,  6,  4,  2,  2,  2,  4,  4,  1,  4,  5,  5,  3,  4,  3,  4,  5,  4,  7,  3,  4,  4,  5,  4,  4,  6,  5,  2,  4,  2,  5,  6,  2,  6,  7,
            6,  3,  6,  3,  5,  7,  6,  7,  6,  5,  5,  6,  4,  5,  6,  6,  4,  5,  6,  6,  6,  3,  6,  7,  7,  6,  7,  7,  6,  7,  6,  7,  6,  6,  7,  7,  4,  6,  6,  7,  4,  7,  7,  7,  7,  5,  7,  7,
            0,  2,  2,  0,  0,  2,  0,  0,  0,  0,  1,  1,  0,  0,  0,  1,  0,  1,  0,  2,  1,  1,  1,  1,  0,  3,  2,  2,  2,  3,  0,  1,  0,  1,  2,  3,  1,  3,  1,  1,  1,  1,  1,  3,  3,  4,  4,  2,
            0,  4,  2,  3,  2,  4,  0,  3,  4,  1,  2,  3,  4,  3,  1,  1,  2,  1,  1,  3,  4,  5,  4,  3,  2,  4,  6,  4,  4,  5,  0,  6,  5,  3,  3,  5,  4,  5,  3,  4,  3,  5,  2,  3,  4,  5,  6,  4,
            4,  5,  6,  5,  4,  5,  5,  6,  6,  5,  4,  5,  5,  5,  6,  7,  4,  5,  3,  5,  5,  5,  6,  5,  4,  7,  6,  6,  4,  6,  6,  7,  6,  5,  6,  6,  6,  5,  6,  7,  6,  5,  6,  7,  6,  7,  6,  5,
            5,  7,  6,  7,  4,  7,  6,  7,  7,  6,  6,  7,  7,  5,  7,  7,  7,  6,  7,  7,  6,  7,  7,  7,  0,  1,  0,  1,  0,  0,  1,  3,  0,  1,  1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  2,  1,  2,  3,
            0,  1,  1,  1,  1,  1,  1,  3,  2,  1,  2,  2,  0,  2,  2,  1,  0,  1,  0,  1,  4,  1,  2,  4,  1,  1,  2,  3,  2,  1,  2,  3,  3,  2,  2,  3,  3,  3,  2,  2,  0,  2,  0,  3,  4,  4,  2,  5,
            2,  2,  4,  3,  4,  5,  3,  3,  4,  5,  3,  3,  4,  3,  3,  3,  2,  3,  2,  3,  6,  5,  3,  6,  3,  3,  4,  3,  5,  5,  4,  4,  5,  7,  4,  5,  4,  4,  6,  5,  2,  4,  2,  4,  6,  5,  5,  6,
            4,  3,  5,  6,  6,  5,  6,  7,  6,  7,  4,  5,  4,  5,  6,  6,  2,  5,  2,  5,  6,  6,  6,  7,  7,  3,  7,  7,  7,  5,  7,  7,  7,  7,  6,  5,  6,  6,  6,  7,  4,  6,  2,  6,  7,  7,  6,  7,
            0,  0,  1,  0,  2,  0,  0,  2,  0,  1,  0,  0,  0,  1,  2,  0,  0,  1,  0,  3,  0,  0,  0,  0,  0,  1,  2,  1,  2,  1,  0,  3,  0,  1,  2,  1,  0,  2,  4,  1,  2,  1,  1,  3,  1,  3,  1,  5,
            0,  1,  2,  2,  2,  2,  0,  4,  0,  1,  2,  3,  1,  3,  4,  1,  2,  1,  2,  3,  2,  5,  2,  5,  1,  3,  2,  3,  3,  4,  4,  5,  1,  2,  5,  3,  3,  3,  5,  3,  3,  1,  3,  3,  4,  5,  2,  5,
            4,  4,  2,  4,  4,  6,  6,  6,  4,  3,  6,  4,  4,  4,  6,  3,  5,  5,  5,  6,  5,  7,  3,  7,  4,  6,  6,  7,  4,  7,  6,  7,  6,  4,  6,  5,  4,  5,  6,  7,  6,  5,  6,  7,  6,  7,  5,  7,
            6,  7,  6,  7,  4,  7,  6,  7,  6,  7,  7,  6,  5,  5,  7,  7,  7,  5,  7,  7,  7,  7,  6,  7,  0,  1,  0,  2,  0,  1,  0,  1,  0,  1,  2,  1,  0,  0,  2,  2,  0,  0,  0,  0,  0,  0,  2,  0,
            0,  1,  2,  3,  1,  1,  1,  1,  0,  1,  2,  3,  0,  1,  4,  4,  0,  0,  0,  1,  0,  1,  2,  2,  2,  3,  3,  3,  2,  3,  3,  3,  1,  5,  3,  3,  0,  3,  4,  5,  0,  2,  0,  2,  3,  2,  2,  5,
            3,  3,  4,  3,  2,  5,  4,  3,  3,  6,  4,  5,  1,  5,  6,  6,  2,  3,  1,  3,  4,  3,  6,  5,  3,  4,  4,  7,  3,  5,  5,  7,  4,  7,  5,  5,  2,  5,  6,  7,  2,  5,  2,  4,  4,  4,  6,  6,
            4,  6,  5,  7,  4,  5,  6,  7,  6,  7,  6,  6,  4,  6,  6,  7,  5,  5,  2,  5,  4,  5,  6,  7,  5,  7,  6,  7,  5,  5,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  6,  6,  2,  7,  6,  5,  7,  7,
            0,  0,  0,  0,  2,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  1,  0,  1,  2,  1,  1,  5,  0,  3,  0,  1,  2,  3,  2,  0,  0,  1,  0,  0,  2,  2,  3,  3,  1,  2,  1,  1,  3,  1,  2,  5,  2,  4,
            2,  1,  2,  3,  4,  2,  0,  2,  0,  1,  2,  3,  4,  3,  2,  3,  2,  1,  4,  1,  3,  5,  5,  5,  4,  2,  4,  4,  4,  5,  3,  3,  4,  2,  2,  3,  4,  5,  3,  3,  3,  4,  5,  3,  4,  7,  5,  5,
            4,  3,  4,  5,  4,  5,  6,  5,  4,  5,  4,  4,  4,  5,  4,  4,  3,  5,  6,  3,  5,  7,  6,  5,  5,  5,  5,  6,  4,  6,  6,  6,  4,  6,  6,  5,  6,  5,  4,  7,  4,  5,  6,  3,  6,  7,  6,  7,
            7,  5,  6,  7,  7,  7,  6,  7,  5,  7,  6,  5,  6,  6,  6,  7,  6,  5,  7,  7,  7,  7,  7,  7,  0,  0,  1,  2,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  2,  0,  0,  0,  0,  0,  0,  0,  2,  1,
            0,  1,  2,  3,  1,  1,  1,  1,  1,  1,  2,  3,  0,  1,  3,  1,  0,  1,  0,  1,  0,  0,  2,  2,  1,  1,  2,  3,  2,  2,  2,  1,  2,  1,  2,  3,  3,  1,  4,  3,  1,  1,  2,  3,  4,  1,  2,  3,
            4,  2,  4,  3,  2,  5,  4,  3,  3,  1,  3,  5,  4,  2,  5,  3,  2,  3,  2,  4,  4,  2,  3,  4,  6,  3,  4,  5,  3,  5,  5,  7,  4,  7,  4,  5,  4,  4,  5,  6,  2,  3,  2,  5,  4,  3,  6,  6,
            6,  3,  4,  7,  4,  5,  6,  7,  5,  7,  5,  5,  5,  5,  6,  7,  2,  6,  2,  6,  4,  3,  6,  7,  6,  5,  7,  7,  5,  7,  7,  7,  7,  7,  7,  6,  6,  5,  6,  7,  7,  7,  5,  7,  6,  6,  6,  7,
            0,  1,  0,  0,  0,  1,  0,  1,  0,  0,  2,  0,  2,  0,  4,  0,  0,  0,  0,  0,  0,  5,  0,  0,  0,  1,  2,  2,  1,  3,  0,  1,  0,  1,  2,  1,  4,  1,  4,  2,  0,  1,  1,  1,  1,  5,  1,  3,
            0,  4,  2,  3,  2,  3,  2,  2,  0,  2,  2,  3,  4,  1,  4,  3,  2,  1,  2,  1,  3,  5,  2,  5,  1,  5,  3,  3,  2,  4,  2,  4,  0,  4,  2,  4,  6,  2,  6,  5,  2,  1,  3,  2,  4,  6,  3,  5,
            2,  5,  4,  4,  3,  5,  6,  5,  4,  5,  5,  5,  6,  4,  6,  7,  3,  3,  6,  3,  5,  7,  4,  7,  3,  6,  5,  7,  4,  5,  6,  6,  4,  6,  6,  6,  6,  5,  6,  7,  5,  5,  6,  3,  6,  7,  5,  7,
            6,  7,  7,  7,  4,  6,  7,  7,  4,  7,  6,  7,  7,  7,  7,  7,  6,  5,  7,  3,  7,  7,  6,  7,  0,  1,  0,  0,  0,  1,  1,  1,  1,  0,  0,  2,  0,  1,  1,  1,  0,  1,  1,  1,  0,  0,  0,  0,
            0,  1,  2,  1,  2,  3,  3,  1,  2,  1,  1,  3,  1,  1,  4,  2,  0,  2,  2,  1,  0,  2,  2,  2,  4,  2,  2,  3,  4,  3,  4,  4,  4,  1,  2,  3,  2,  1,  4,  3,  2,  3,  2,  2,  2,  3,  2,  3,
            5,  3,  4,  6,  4,  5,  5,  5,  4,  1,  3,  3,  3,  5,  5,  3,  3,  3,  6,  3,  4,  4,  6,  4,  6,  4,  4,  7,  5,  5,  6,  7,  5,  7,  4,  5,  4,  5,  6,  4,  4,  4,  6,  3,  4,  5,  6,  5,
            6,  5,  5,  7,  6,  5,  6,  7,  7,  7,  6,  7,  4,  5,  6,  7,  5,  5,  6,  4,  4,  7,  6,  6,  6,  6,  6,  7,  7,  6,  6,  7,  7,  7,  7,  7,  7,  6,  7,  7,  6,  7,  7,  7,  5,  7,  6,  7,
            0,  0,  1,  2,  2,  0,  0,  0,  0,  1,  0,  0,  4,  1,  2,  1,  0,  1,  2,  1,  1,  2,  0,  3,  1,  1,  2,  3,  2,  1,  1,  1,  0,  2,  0,  1,  4,  4,  3,  3,  0,  1,  2,  1,  4,  3,  0,  3,
            4,  1,  2,  3,  4,  2,  2,  2,  0,  3,  0,  2,  4,  4,  4,  4,  1,  3,  2,  3,  4,  5,  1,  3,  5,  1,  3,  3,  4,  3,  2,  2,  3,  4,  2,  3,  6,  5,  4,  5,  2,  4,  3,  3,  4,  5,  2,  6,
            6,  3,  4,  5,  4,  3,  3,  3,  4,  5,  2,  4,  6,  5,  4,  6,  4,  5,  4,  3,  6,  5,  4,  7,  7,  5,  4,  5,  5,  5,  6,  5,  4,  5,  2,  5,  6,  6,  6,  7,  4,  6,  5,  4,  7,  7,  5,  7,
            7,  5,  7,  7,  7,  7,  6,  7,  4,  6,  6,  6,  6,  7,  6,  7,  7,  7,  6,  6,  7,  7,  7,  7,  0,  1,  0,  2,  0,  1,  0,  0,  0,  1,  0,  3,  0,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,
            0,  1,  2,  3,  0,  2,  2,  1,  0,  1,  2,  3,  1,  1,  1,  3,  0,  1,  2,  1,  0,  1,  0,  1,  0,  2,  2,  3,  0,  3,  2,  1,  2,  1,  3,  3,  2,  1,  2,  5,  1,  1,  2,  1,  0,  4,  1,  2,
            0,  3,  4,  4,  1,  4,  5,  1,  2,  1,  4,  3,  2,  2,  3,  5,  2,  1,  3,  1,  1,  5,  2,  3,  6,  4,  4,  5,  3,  4,  6,  6,  5,  4,  5,  5,  5,  5,  5,  7,  2,  3,  4,  2,  4,  5,  6,  3,
            6,  6,  4,  7,  4,  5,  6,  7,  6,  7,  6,  7,  7,  5,  6,  7,  5,  3,  6,  3,  4,  6,  6,  4,  6,  7,  5,  7,  4,  5,  7,  7,  7,  7,  7,  7,  7,  5,  6,  7,  7,  3,  6,  7,  6,  7,  6,  7,
            0,  1,  0,  1,  0,  3,  1,  1,  0,  0,  0,  0,  4,  0,  2,  0,  0,  1,  1,  0,  2,  1,  0,  3,  1,  1,  2,  3,  0,  3,  2,  1,  0,  1,  0,  2,  4,  1,  4,  1,  0,  2,  2,  1,  2,  1,  0,  3,
            3,  1,  3,  3,  1,  3,  4,  2,  0,  2,  1,  3,  4,  2,  4,  2,  0,  3,  2,  2,  2,  4,  3,  6,  4,  1,  4,  5,  3,  5,  6,  4,  2,  5,  2,  4,  5,  3,  6,  4,  3,  4,  6,  3,  3,  5,  4,  7,
            5,  4,  5,  5,  4,  5,  6,  5,  4,  5,  2,  5,  6,  4,  6,  5,  4,  5,  6,  3,  4,  6,  5,  7,  6,  5,  6,  5,  6,  5,  6,  7,  4,  6,  2,  6,  6,  5,  6,  7,  4,  6,  6,  5,  4,  7,  6,  7,
            7,  5,  7,  6,  7,  6,  7,  7,  7,  7,  7,  7,  6,  7,  7,  7,  6,  7,  7,  7,  5,  7,  6,  7,  0,  0,  1,  0,  0,  0,  0,  1,  0,  0,  0,  1,  1,  1,  1,  3,  0,  1,  0,  1,  0,  0,  0,  0,
            0,  1,  2,  1,  0,  3,  1,  2,  1,  1,  0,  3,  2,  1,  2,  5,  1,  1,  2,  3,  2,  1,  0,  1,  0,  2,  2,  2,  0,  3,  2,  3,  2,  1,  1,  3,  3,  5,  3,  5,  2,  3,  2,  3,  4,  2,  1,  3,
            6,  3,  2,  3,  4,  4,  2,  4,  2,  5,  2,  3,  4,  5,  4,  5,  2,  3,  5,  3,  4,  3,  3,  3,  6,  5,  4,  3,  4,  5,  3,  5,  3,  5,  2,  4,  6,  5,  5,  7,  3,  3,  6,  4,  4,  5,  4,  4,
            6,  6,  6,  6,  4,  5,  6,  7,  4,  5,  3,  7,  6,  5,  6,  7,  4,  4,  6,  6,  5,  5,  6,  6,  7,  7,  6,  7,  5,  6,  6,  7,  6,  6,  6,  7,  7,  7,  7,  7,  5,  7,  7,  7,  7,  5,  6,  7,
            0,  1,  0,  3,  0,  1,  0,  0,  0,  1,  0,  0,  1,  0,  2,  0,  0,  0,  1,  1,  0,  0,  0,  2,  1,  1,  1,  3,  3,  2,  2,  1,  0,  1,  0,  3,  2,  1,  2,  3,  0,  1,  2,  1,  1,  1,  0,  3,
            2,  1,  2,  3,  3,  3,  3,  3,  2,  1,  2,  3,  4,  3,  2,  4,  0,  1,  2,  2,  2,  1,  1,  3,  3,  5,  3,  5,  4,  3,  4,  3,  3,  2,  2,  3,  4,  4,  5,  5,  0,  2,  4,  3,  2,  2,  2,  3,
            4,  7,  4,  5,  4,  3,  5,  6,  4,  3,  2,  5,  4,  6,  6,  6,  4,  3,  4,  3,  4,  5,  4,  5,  4,  7,  5,  5,  5,  5,  6,  7,  5,  4,  3,  6,  6,  7,  6,  6,  4,  6,  6,  4,  4,  5,  6,  7,
            5,  7,  7,  5,  6,  5,  6,  7,  6,  5,  5,  6,  6,  7,  6,  7,  7,  7,  6,  6,  6,  7,  7,  7,  0,  1,  2,  0,  0,  0,  2,  0,  0,  0,  0,  0,  0,  0,  0,  2,  0,  1,  1,  0,  0,  1,  0,  1,
            0,  2,  2,  1,  0,  1,  4,  1,  0,  1,  0,  3,  3,  1,  1,  5,  1,  1,  2,  3,  1,  1,  0,  1,  0,  3,  2,  1,  0,  2,  4,  1,  0,  1,  0,  3,  4,  5,  2,  5,  2,  1,  2,  3,  2,  4,  5,  4,
            0,  4,  2,  4,  1,  4,  6,  4,  2,  2,  1,  5,  4,  5,  2,  5,  3,  3,  3,  7,  2,  5,  6,  6,  3,  5,  4,  6,  4,  5,  6,  6,  2,  3,  2,  7,  5,  5,  3,  7,  3,  3,  4,  7,  3,  5,  6,  7,
            6,  6,  6,  6,  4,  5,  6,  7,  2,  5,  6,  7,  6,  7,  6,  7,  5,  3,  5,  7,  3,  6,  6,  7,  6,  7,  6,  7,  4,  7,  7,  7,  7,  5,  7,  7,  7,  7,  7,  7,  6,  4,  6,  7,  4,  7,  7,  7,
            0,  0,  1,  3,  0,  1,  0,  2,  0,  0,  0,  1,  0,  2,  2,  0,  0,  0,  0,  0,  2,  0,  0,  0,  1,  1,  2,  3,  1,  3,  1,  3,  1,  1,  1,  1,  0,  3,  2,  3,  0,  2,  2,  2,  2,  3,  0,  1,
            4,  1,  3,  3,  2,  3,  3,  3,  2,  1,  2,  1,  3,  4,  5,  4,  0,  3,  2,  3,  2,  5,  1,  3,  4,  1,  4,  4,  3,  5,  4,  5,  3,  5,  2,  2,  4,  5,  6,  5,  0,  4,  2,  3,  4,  5,  3,  3,
            5,  6,  5,  5,  4,  5,  6,  7,  4,  5,  4,  3,  5,  5,  6,  5,  2,  5,  4,  5,  4,  5,  4,  6,  6,  7,  6,  5,  6,  5,  6,  7,  5,  5,  6,  3,  6,  7,  6,  6,  4,  6,  4,  6,  4,  6,  6,  7,
            7,  7,  7,  5,  7,  6,  7,  7,  6,  6,  7,  4,  6,  7,  6,  7,  4,  7,  4,  7,  5,  7,  6,  7,  0,  0,  0,  0,  0,  0,  2,  0,  0,  1,  0,  1,  0,  0,  0,  3,  0,  1,  0,  3,  1,  0,  0,  1,
            0,  1,  2,  1,  0,  1,  4,  1,  0,  1,  2,  3,  1,  2,  2,  5,  1,  1,  1,  3,  2,  1,  1,  1,  4,  1,  2,  2,  4,  2,  4,  1,  2,  4,  2,  3,  2,  3,  2,  5,  4,  1,  2,  3,  3,  1,  2,  2,
            4,  2,  2,  3,  4,  3,  4,  2,  2,  5,  2,  3,  4,  5,  3,  6,  5,  2,  4,  4,  4,  2,  3,  3,  4,  5,  3,  5,  4,  4,  6,  3,  2,  5,  3,  4,  4,  5,  5,  7,  6,  3,  5,  7,  4,  5,  4,  5,
            5,  7,  6,  5,  4,  5,  6,  4,  3,  6,  5,  6,  4,  7,  6,  7,  6,  5,  6,  7,  5,  5,  6,  7,  7,  7,  6,  6,  6,  6,  6,  7,  6,  7,  6,  7,  7,  7,  7,  7,  7,  5,  7,  7,  7,  5,  7,  7,
            0,  1,  2,  0,  0,  1,  2,  3,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  2,  0,  0,  1,  0,  1,  0,  1,  2,  1,  2,  1,  3,  3,  0,  1,  0,  1,  0,  1,  2,  1,  0,  1,  2,  2,  2,  2,  0,  2,
            0,  1,  2,  3,  3,  1,  4,  6,  1,  1,  1,  1,  1,  3,  2,  2,  0,  1,  2,  2,  2,  3,  2,  3,  1,  2,  4,  3,  5,  4,  5,  7,  2,  3,  2,  3,  4,  5,  2,  3,  1,  3,  4,  3,  2,  4,  2,  4,
            2,  4,  5,  3,  6,  5,  5,  7,  5,  3,  2,  3,  4,  6,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  3,  7,  5,  5,  6,  5,  6,  7,  6,  5,  3,  5,  6,  7,  6,  7,  6,  6,  4,  5,  4,  6,  6,  7,
            4,  7,  7,  5,  7,  5,  7,  7,  7,  5,  5,  7,  7,  7,  6,  7,  6,  7,  7,  7,  4,  7,  6,  7,  0,  0,  2,  2,  0,  0,  1,  0,  0,  0,  2,  0,  0,  4,  0,  0,  0,  1,  0,  3,  0,  1,  0,  1,
            0,  1,  2,  3,  4,  1,  4,  1,  0,  1,  2,  1,  0,  5,  0,  2,  0,  1,  0,  3,  1,  3,  3,  1,  1,  5,  4,  3,  4,  1,  4,  2,  0,  1,  6,  1,  3,  5,  3,  5,  2,  1,  3,  3,  3,  3,  4,  1,
            2,  6,  6,  4,  4,  2,  4,  2,  2,  2,  6,  2,  4,  5,  5,  5,  3,  3,  4,  7,  4,  5,  5,  1,  4,  7,  6,  5,  6,  5,  6,  4,  2,  3,  6,  2,  4,  6,  6,  7,  4,  5,  5,  7,  5,  5,  6,  6,
            4,  7,  6,  6,  6,  6,  6,  5,  2,  4,  6,  3,  5,  7,  6,  7,  5,  5,  5,  7,  6,  5,  6,  7,  7,  7,  7,  7,  7,  7,  6,  7,  3,  5,  7,  7,  6,  7,  6,  7,  6,  6,  6,  7,  7,  5,  7,  7,
            0,  0,  0,  2,  1,  1,  2,  1,  1,  1,  1,  1,  2,  0,  0,  3,  0,  0,  2,  0,  0,  0,  1,  0,  0,  1,  0,  3,  2,  1,  3,  3,  2,  1,  2,  1,  4,  2,  2,  3,  0,  1,  2,  1,  2,  1,  2,  2,
            4,  1,  0,  4,  3,  4,  4,  3,  3,  1,  2,  1,  4,  3,  2,  4,  4,  1,  2,  2,  2,  2,  2,  2,  4,  3,  1,  5,  4,  5,  4,  7,  4,  3,  4,  3,  4,  5,  5,  5,  5,  2,  3,  3,  4,  3,  6,  3,
            4,  5,  2,  5,  4,  5,  5,  7,  5,  3,  5,  3,  5,  5,  6,  6,  6,  3,  4,  5,  4,  5,  6,  4,  5,  6,  2,  7,  6,  5,  6,  7,  6,  3,  6,  3,  6,  7,  6,  7,  6,  4,  4,  5,  4,  6,  6,  5,
            7,  7,  3,  7,  6,  5,  7,  7,  7,  7,  7,  4,  7,  7,  7,  7,  7,  5,  4,  6,  7,  7,  6,  7,  0,  1,  0,  0,  1,  1,  2,  0,  0,  0,  2,  0,  0,  0,  0,  1,  0,  1,  0,  2,  0,  3,  0,  1,
            0,  1,  2,  1,  2,  1,  4,  3,  0,  1,  2,  1,  0,  1,  0,  3,  0,  1,  1,  3,  0,  3,  0,  1,  3,  1,  2,  3,  4,  2,  4,  4,  0,  3,  2,  2,  1,  2,  1,  3,  1,  1,  2,  3,  1,  3,  1,  1,
            4,  2,  2,  3,  4,  4,  5,  5,  1,  4,  3,  3,  4,  3,  2,  3,  6,  2,  2,  3,  2,  4,  2,  5,  4,  4,  3,  3,  6,  5,  6,  5,  2,  5,  6,  3,  4,  5,  4,  4,  6,  5,  4,  3,  3,  5,  3,  7,
            5,  5,  5,  6,  6,  6,  6,  6,  4,  6,  6,  4,  4,  6,  6,  6,  6,  5,  5,  7,  4,  5,  5,  7,  6,  7,  6,  7,  7,  7,  6,  7,  4,  7,  6,  6,  7,  7,  6,  7,  7,  7,  7,  7,  5,  5,  7,  7,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  0,  1,  0,  0,  0,  1,  0,  1,  0,  1,  2,  0,  0,  1,  0,  3,  2,  1,  1,  3,  0,  1,  1,  1,  1,  1,  1,  2,  0,  1,  2,  3,  0,  2,  2,  1,
            2,  1,  2,  3,  4,  1,  2,  5,  2,  1,  2,  1,  2,  4,  2,  3,  0,  3,  2,  4,  0,  3,  4,  2,  2,  2,  2,  5,  5,  1,  3,  5,  3,  3,  3,  3,  3,  5,  3,  3,  1,  4,  4,  5,  3,  3,  6,  4,
            4,  4,  4,  6,  6,  3,  6,  7,  5,  3,  4,  3,  4,  5,  4,  7,  3,  4,  4,  5,  4,  4,  6,  5,  4,  7,  6,  7,  6,  5,  6,  7,  6,  3,  6,  3,  5,  7,  6,  7,  6,  5,  5,  6,  4,  5,  6,  6,
            7,  7,  7,  7,  7,  5,  7,  7,  7,  6,  7,  7,  6,  7,  6,  7,  6,  6,  7,  7,  4,  6,  6,  7,  0,  1,  0,  2,  3,  2,  1,  1,  0,  2,  2,  0,  0,  2,  0,  0,  0,  0,  1,  1,  0,  0,  0,  1,
            0,  1,  0,  3,  4,  4,  4,  2,  0,  3,  2,  2,  2,  3,  0,  1,  0,  1,  2,  3,  1,  3,  1,  1,  1,  5,  1,  3,  4,  5,  4,  3,  0,  4,  2,  3,  2,  4,  0,  3,  4,  1,  2,  3,  4,  3,  1,  1,
            2,  5,  1,  4,  4,  5,  6,  4,  2,  4,  6,  4,  4,  5,  0,  6,  5,  3,  3,  5,  4,  5,  3,  4,  3,  5,  2,  5,  5,  5,  6,  5,  4,  5,  6,  5,  4,  5,  5,  6,  6,  5,  4,  5,  5,  5,  6,  7,
            4,  5,  6,  7,  6,  7,  6,  7,  4,  7,  6,  6,  4,  6,  6,  7,  6,  5,  6,  6,  6,  5,  6,  7,  7,  6,  7,  7,  6,  7,  7,  7,  5,  7,  6,  7,  4,  7,  6,  7,  7,  6,  6,  7,  7,  5,  7,  7,
            0,  0,  0,  1,  1,  1,  2,  3,  0,  1,  0,  1,  0,  0,  1,  3,  0,  1,  1,  0,  0,  1,  0,  0,  0,  1,  0,  3,  2,  1,  2,  4,  0,  1,  1,  1,  1,  1,  1,  3,  2,  1,  2,  2,  0,  2,  2,  1,
            0,  1,  0,  3,  4,  4,  3,  5,  1,  1,  2,  3,  2,  1,  2,  3,  3,  2,  2,  3,  3,  3,  2,  2,  2,  3,  2,  3,  4,  5,  4,  5,  2,  2,  4,  3,  4,  5,  3,  3,  4,  5,  3,  3,  4,  3,  3,  3,
            2,  4,  2,  4,  6,  5,  5,  6,  3,  3,  4,  3,  5,  5,  4,  4,  5,  7,  4,  5,  4,  4,  6,  5,  2,  5,  2,  5,  6,  6,  6,  7,  4,  3,  5,  6,  6,  5,  6,  7,  6,  7,  4,  5,  4,  5,  6,  6,
            6,  6,  3,  6,  7,  7,  6,  7,  7,  3,  7,  7,  7,  5,  7,  7,  7,  7,  6,  5,  6,  6,  6,  7,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  2,  0,  0,  2,  0,  1,  0,  0,  0,  1,  2,  0,
            2,  1,  1,  3,  1,  1,  2,  5,  0,  1,  2,  1,  2,  1,  0,  3,  0,  1,  2,  1,  0,  2,  4,  1,  3,  1,  2,  3,  2,  3,  2,  5,  0,  1,  2,  2,  2,  2,  0,  4,  0,  1,  2,  3,  1,  3,  4,  1,
            4,  2,  2,  3,  4,  5,  2,  5,  1,  3,  2,  3,  3,  4,  4,  5,  1,  2,  5,  3,  3,  3,  5,  3,  5,  5,  3,  6,  5,  5,  3,  6,  4,  4,  2,  4,  4,  6,  6,  6,  4,  3,  6,  4,  4,  4,  6,  3,
            6,  5,  5,  7,  6,  7,  5,  7,  4,  6,  6,  7,  4,  7,  6,  7,  6,  4,  6,  5,  4,  5,  6,  7,  7,  5,  7,  7,  7,  7,  6,  7,  6,  7,  6,  7,  4,  7,  6,  7,  6,  7,  7,  6,  5,  5,  7,  7,
        };
    }
}
