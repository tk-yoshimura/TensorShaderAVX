using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.ConnectionDense;

namespace TensorShaderTest.Operators.ConnectionDense {
    [TestClass]
    public class TransposeDenseTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach(int inchannels in new int[]{ 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach(int outchannels in new int[]{ 7, 13 }) {
                        float[] yval = (new float[outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                        float[] wval = (new float[inchannels * outchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                        Map0D y = new Map0D(outchannels, batch, yval);
                        Filter0D w = new Filter0D(inchannels, outchannels, 1, wval);

                        Map0D x = Reference(y, w);

                        OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map0D(outchannels, batch), yval);
                        OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels, outchannels), wval);

                        OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map0D(inchannels, batch));

                        TransposeDense ope = new TransposeDense(outchannels, inchannels, batch);

                        ope.Execute(y_tensor, w_tensor, x_tensor);

                        float[] x_expect = x.ToArray();
                        float[] x_actual = x_tensor.State;

                        CollectionAssert.AreEqual(yval, y_tensor.State);
                        CollectionAssert.AreEqual(wval, w_tensor.State);

                        AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{batch}");

                        Console.WriteLine($"pass: {inchannels},{outchannels},{batch}");
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inchannels = 1024, outchannels = 512;

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map0D(outchannels));
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels, outchannels));

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map0D(inchannels));

            TransposeDense ope = new TransposeDense(outchannels, inchannels);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            ope.Execute(y_tensor, w_tensor, x_tensor);
            ope.Execute(y_tensor, w_tensor, x_tensor);
            ope.Execute(y_tensor, w_tensor, x_tensor);
            ope.Execute(y_tensor, w_tensor, x_tensor);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds / 4} msec");
        }

        public static Map0D Reference(Map0D y, Filter0D w) {
            int outchannels = y.Channels, inchannels = w.InChannels, batch = y.Batch;

            Map0D x = new Map0D(inchannels, batch);

            for (int th = 0; th < batch; th++) {
                for (int inch = 0; inch < inchannels; inch++) {
                    double sum = 0;

                    for (int outch = 0; outch < outchannels; outch++) {
                        sum += y[outch, th] * w[inch, outch, 0];
                    }

                    x[inch, th] = sum;
                }
            }

            return x;
        }

        public static Map0D OptimizedReference(Map0D y, Filter0D w) {
            int outchannels = y.Channels, inchannels = w.InChannels, batch = y.Batch;

            Map0D x = new Map0D(inchannels, batch);

            for (int th = 0; th < batch; th++) {
                for (int inch = 0; inch < inchannels; inch++) {
                    double sum = 0;

                    int inmap_idx = outchannels * th;
                    int outmap_idx = inch + inchannels * th;
                    int kernel_idx = inch;

                    for (int outch = 0; outch < outchannels; outch++) {
                        sum += y[inmap_idx] * w[kernel_idx];

                        inmap_idx++;
                        kernel_idx += inchannels;
                    }

                    x[outmap_idx] = sum;
                }
            }

            return x;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 7, outchannels = 11, batch = 2;

            float[] yval = (new float[outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[outchannels * inchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map0D y = new Map0D(outchannels, batch, yval);
            Filter0D w = new Filter0D(inchannels, outchannels, 1, wval);

            Map0D x = Reference(y, w);

            float[] x_expect = {
                1.4850e-03f, 1.4300e-03f, 1.3750e-03f, 1.3200e-03f, 1.2650e-03f,
                1.2100e-03f, 1.1550e-03f, 6.4460e-03f, 6.2700e-03f, 6.0940e-03f,
                5.9180e-03f, 5.7420e-03f, 5.5660e-03f, 5.3900e-03f
            };

            float[] x_actual = x.ToArray();

            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{batch}");
        }

        [TestMethod]
        public void OptimizeTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach (int outchannels in new int[] { 7, 13 }) {
                        float[] yval = (new float[outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                        float[] wval = (new float[inchannels * outchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                        Map0D y = new Map0D(outchannels, batch, yval);
                        Filter0D w = new Filter0D(inchannels, outchannels, 1, wval);

                        Map0D x = Reference(y, w);
                        Map0D x_optimized = OptimizedReference(y, w);

                        float[] x_expect = x.ToArray();
                        float[] x_actual = x_optimized.ToArray();

                        AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{batch}");

                        Console.WriteLine($"pass: {inchannels},{outchannels},{batch}");
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }
    }
}
