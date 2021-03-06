using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.QuaternionConvolution;

namespace TensorShaderTest.Operators.Quaternion {
    [TestClass]
    public class QuaternionKernelProductDenseTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2, 3 }) {
                foreach (int inchannels in new int[] { 4, 8, 12 }) {
                    foreach (int outchannels in new int[] { 4, 8, 12 }) {
                        float[] xval = (new float[inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                        float[] yval = (new float[outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                        Quaternion[] xcval = (new Quaternion[xval.Length / 4])
                            .Select((_, idx) => new Quaternion(xval[idx * 4], xval[idx * 4 + 1], xval[idx * 4 + 2], xval[idx * 4 + 3])).ToArray();

                        Quaternion[] ycval = (new Quaternion[yval.Length / 4])
                            .Select((_, idx) => new Quaternion(yval[idx * 4], yval[idx * 4 + 1], yval[idx * 4 + 2], yval[idx * 4 + 3])).ToArray();

                        QuaternionMap0D x = new QuaternionMap0D(inchannels / 4, batch, xcval);
                        QuaternionMap0D y = new QuaternionMap0D(outchannels / 4, batch, ycval);

                        QuaternionFilter0D gw = Reference(x, y);

                        OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map0D(inchannels, batch), xval);
                        OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map0D(outchannels, batch), yval);

                        OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels, outchannels / 4));

                        QuaternionKernelProductDense ope = new QuaternionKernelProductDense(inchannels, outchannels, transpose: false, batch);

                        ope.Execute(x_tensor, y_tensor, gw_tensor);

                        float[] gw_expect = gw.ToArray();
                        float[] gw_actual = gw_tensor.State;

                        CollectionAssert.AreEqual(xval, x_tensor.State);
                        CollectionAssert.AreEqual(yval, y_tensor.State);

                        AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{batch}");

                        Console.WriteLine($"pass: {inchannels},{outchannels},{batch}");

                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void OverflowTest() {
            foreach(bool transpose in new bool[] { false, true }) { 
                foreach (int batch in new int[] { 1, 2, 3 }) {
                    foreach (int inchannels in new int[] { 4, 8, 12 }) {
                        foreach (int outchannels in new int[] { 4, 8, 12 }) {
                            float[] xval = (new float[inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                            float[] yval = (new float[outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map0D(inchannels, batch), xval);
                            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map0D(outchannels, batch), yval);

                            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels, outchannels / 4));

                            QuaternionKernelProductDense ope = new QuaternionKernelProductDense(inchannels, outchannels, transpose, batch);

                            ope.Execute(x_tensor, y_tensor, gw_tensor);

                            CollectionAssert.AreEqual(xval, x_tensor.State);
                            CollectionAssert.AreEqual(yval, y_tensor.State);

                            gw_tensor.CheckOverflow();

                            Console.WriteLine($"pass: {inchannels},{outchannels},{batch},{transpose}");

                        }
                    }
                }
            }
        }

        [TestMethod]
        public void SpeedTest() {
            int inchannels = 32, outchannels = 32;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map0D(inchannels));
            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map0D(outchannels));

            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels, outchannels / 4));

            QuaternionKernelProductDense ope = new QuaternionKernelProductDense(inchannels, outchannels);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            ope.Execute(x_tensor, y_tensor, gw_tensor);
            ope.Execute(x_tensor, y_tensor, gw_tensor);
            ope.Execute(x_tensor, y_tensor, gw_tensor);
            ope.Execute(x_tensor, y_tensor, gw_tensor);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds / 4} msec");
        }

        public static QuaternionFilter0D Reference(QuaternionMap0D x, QuaternionMap0D gy) {
            int inchannels = x.Channels, outchannels = gy.Channels, batch = x.Batch;

            QuaternionFilter0D w = new QuaternionFilter0D(inchannels, outchannels);

            for (int inch, outch = 0; outch < outchannels; outch++) {
                for (inch = 0; inch < inchannels; inch++) {
                    Quaternion sum = 0;

                    for (int th = 0; th < batch; th++) {
                        sum += Quaternion.MulGrad(gy[outch, th], x[inch, th]);
                    }

                    w[inch, outch] = sum;
                }

            }

            return w;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 8, outchannels = 12, batch = 3;

            float[] xval = (new float[inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Quaternion[] xcval = (new Quaternion[xval.Length / 4])
                .Select((_, idx) => new Quaternion(xval[idx * 4], xval[idx * 4 + 1], xval[idx * 4 + 2], xval[idx * 4 + 3])).ToArray();

            Quaternion[] ycval = (new Quaternion[yval.Length / 4])
                .Select((_, idx) => new Quaternion(yval[idx * 4], yval[idx * 4 + 1], yval[idx * 4 + 2], yval[idx * 4 + 3])).ToArray();

            QuaternionMap0D x = new QuaternionMap0D(inchannels / 4, batch, xcval);
            QuaternionMap0D y = new QuaternionMap0D(outchannels / 4, batch, ycval);

            QuaternionFilter0D gw = Reference(x, y);

            float[] gw_expect = {
                1.668000e-03f,  -1.084202e-19f,  -3.720000e-04f,  -1.860000e-04f,  2.700000e-03f,  1.084202e-19f,
                -4.200000e-04f,  -2.100000e-04f,  1.212000e-03f,  -0.000000e+00f,  -3.240000e-04f,  -1.620000e-04f,
                2.052000e-03f,  1.084202e-19f,  -3.720000e-04f,  -1.860000e-04f,  7.560000e-04f,  -0.000000e+00f,
                -2.760000e-04f,  -1.380000e-04f,  1.404000e-03f,  5.421011e-20f,  -3.240000e-04f,  -1.620000e-04f,
            };

            float[] gw_actual = gw.ToArray();

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{batch}");
        }
    }
}
