using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.ComplexConvolution;

namespace TensorShaderTest.Operators.Complex {
    [TestClass]
    public class ComplexTransposeDenseTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2, 3 }) {
                foreach (int inchannels in new int[] { 2, 4, 10, 20 }) {
                    foreach (int outchannels in new int[] { 6, 14 }) {
                        float[] yval = (new float[outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                        float[] wval = (new float[inchannels * outchannels / 2]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                        System.Numerics.Complex[] ycval = (new System.Numerics.Complex[yval.Length / 2])
                            .Select((_, idx) => new System.Numerics.Complex(yval[idx * 2], yval[idx * 2 + 1])).ToArray();

                        System.Numerics.Complex[] wcval = (new System.Numerics.Complex[wval.Length / 2])
                            .Select((_, idx) => new System.Numerics.Complex(wval[idx * 2], wval[idx * 2 + 1])).ToArray();

                        ComplexMap0D y = new ComplexMap0D(outchannels / 2, batch, ycval);
                        ComplexFilter0D w = new ComplexFilter0D(inchannels / 2, outchannels / 2, wcval);

                        ComplexMap0D x = Reference(y, w);

                        OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map0D(outchannels, batch), yval);
                        OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels, outchannels / 2), wval);

                        OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map0D(inchannels, batch));

                        ComplexTransposeDense ope = new ComplexTransposeDense(outchannels, inchannels, gradmode: false, batch);

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
        public void OverflowTest() {
            foreach(bool gradmode in new bool[] { false, true }) { 
                foreach (int batch in new int[] { 1, 2, 3 }) {
                    foreach (int inchannels in new int[] { 2, 4, 10, 20 }) {
                        foreach (int outchannels in new int[] { 6, 14 }) {
                            float[] yval = (new float[outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                            float[] wval = (new float[inchannels * outchannels / 2]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map0D(outchannels, batch), yval);
                            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels, outchannels / 2), wval);

                            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map0D(inchannels, batch));

                            ComplexTransposeDense ope = new ComplexTransposeDense(outchannels, inchannels, gradmode, batch);

                            ope.Execute(y_tensor, w_tensor, x_tensor);

                            CollectionAssert.AreEqual(yval, y_tensor.State);
                            CollectionAssert.AreEqual(wval, w_tensor.State);

                            x_tensor.CheckOverflow();

                            Console.WriteLine($"pass: {inchannels},{outchannels},{batch},{gradmode}");
                        }
                    }
                }
            }
        }

        [TestMethod]
        public void SpeedTest() {
            int inchannels = 32, outchannels = 32;

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map0D(outchannels));
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels, outchannels / 2));

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map0D(inchannels));

            ComplexTransposeDense ope = new ComplexTransposeDense(outchannels, inchannels);

            ope.Execute(y_tensor, w_tensor, x_tensor);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            ope.Execute(y_tensor, w_tensor, x_tensor);
            ope.Execute(y_tensor, w_tensor, x_tensor);
            ope.Execute(y_tensor, w_tensor, x_tensor);
            ope.Execute(y_tensor, w_tensor, x_tensor);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds / 4} msec");
        }

        public static ComplexMap0D Reference(ComplexMap0D y, ComplexFilter0D w) {
            int inchannels = w.InChannels, outchannels = w.OutChannels, batch = y.Batch;

            ComplexMap0D x = new ComplexMap0D(inchannels, batch);

            for (int th = 0; th < batch; th++) {
                for (int outch = 0; outch < outchannels; outch++) {
                    System.Numerics.Complex v = y[outch, th];

                    for (int inch = 0; inch < inchannels; inch++) {
                        x[inch, th] += v * w[inch, outch];
                    }
                }
            }

            return x;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 6, outchannels = 8, batch = 3;

            float[] yval = (new float[outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[outchannels * inchannels / 2]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            System.Numerics.Complex[] ycval = (new System.Numerics.Complex[yval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(yval[idx * 2], yval[idx * 2 + 1])).ToArray();

            System.Numerics.Complex[] wcval = (new System.Numerics.Complex[wval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(wval[idx * 2], wval[idx * 2 + 1])).ToArray();

            ComplexMap0D y = new ComplexMap0D(outchannels / 2, batch, ycval);
            ComplexFilter0D w = new ComplexFilter0D(inchannels / 2, outchannels / 2, wcval);

            ComplexMap0D x = Reference(y, w);

            float[] x_expect = {
                -4.000000e-05f,  2.600000e-04f,  -3.200000e-05f,  2.040000e-04f,  -2.400000e-05f,  1.480000e-04f,
                -8.000000e-06f,  1.124000e-03f,  0.000000e+00f,  9.400000e-04f,  8.000000e-06f,  7.560000e-04f,
                2.400000e-05f,  1.988000e-03f,  3.200000e-05f,  1.676000e-03f,  4.000000e-05f,  1.364000e-03f,
            };

            float[] x_actual = x.ToArray();

            AssertError.Tolerance(x_expect, x_actual, 1e-9f, 1e-5f, $"mismatch value {inchannels},{outchannels},{batch}");
        }
    }
}
