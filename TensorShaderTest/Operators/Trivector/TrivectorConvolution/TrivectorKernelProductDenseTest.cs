using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.TrivectorConvolution;

namespace TensorShaderTest.Operators.Trivector {
    [TestClass]
    public class TrivectorKernelProductDenseTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2, 3 }) {
                foreach (int inchannels in new int[] { 3, 6, 9, 12 }) {
                    foreach (int outchannels in new int[] { 3, 6, 9, 12 }) {
                        float[] xval = (new float[inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                        float[] yval = (new float[outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();
                        float[] wval = (new float[inchannels * outchannels / 9 * 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                        Trivector[] xcval = (new Trivector[xval.Length / 3])
                            .Select((_, idx) => new Trivector(xval[idx * 3], xval[idx * 3 + 1], xval[idx * 3 + 2])).ToArray();

                        Trivector[] ycval = (new Trivector[yval.Length / 3])
                            .Select((_, idx) => new Trivector(yval[idx * 3], yval[idx * 3 + 1], yval[idx * 3 + 2])).ToArray();

                        Quaternion.Quaternion[] wcval = (new Quaternion.Quaternion[wval.Length / 4])
                            .Select((_, idx) => new Quaternion.Quaternion(wval[idx * 4], wval[idx * 4 + 1], wval[idx * 4 + 2], wval[idx * 4 + 3])).ToArray();

                        TrivectorMap0D x = new TrivectorMap0D(inchannels / 3, batch, xcval);
                        TrivectorMap0D y = new TrivectorMap0D(outchannels / 3, batch, ycval);
                        Quaternion.QuaternionFilter0D w = new Quaternion.QuaternionFilter0D(inchannels / 3, outchannels / 3, wcval);

                        Quaternion.QuaternionFilter0D gw = Reference(x, y, w);

                        OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map0D(inchannels, batch), xval);
                        OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map0D(outchannels, batch), yval);
                        OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels / 3 * 4, outchannels / 3), wval);

                        OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels / 3 * 4, outchannels / 3));

                        TrivectorKernelProductDense ope = new TrivectorKernelProductDense(inchannels, outchannels, transpose: false, batch);

                        ope.Execute(x_tensor, y_tensor, w_tensor, gw_tensor);

                        float[] gw_expect = gw.ToArray();
                        float[] gw_actual = gw_tensor.State;

                        CollectionAssert.AreEqual(xval, x_tensor.State);
                        CollectionAssert.AreEqual(yval, y_tensor.State);
                        CollectionAssert.AreEqual(wval, w_tensor.State);

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
                    foreach (int inchannels in new int[] { 3, 6, 9, 12 }) {
                        foreach (int outchannels in new int[] { 3, 6, 9, 12 }) {
                            float[] xval = (new float[inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                            float[] yval = (new float[outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();
                            float[] wval = (new float[inchannels * outchannels / 9 * 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map0D(inchannels, batch), xval);
                            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map0D(outchannels, batch), yval);
                            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels / 3 * 4, outchannels / 3), wval);

                            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels / 3 * 4, outchannels / 3));

                            TrivectorKernelProductDense ope = new TrivectorKernelProductDense(inchannels, outchannels, transpose, batch);

                            ope.Execute(x_tensor, y_tensor, w_tensor, gw_tensor);

                            CollectionAssert.AreEqual(xval, x_tensor.State);
                            CollectionAssert.AreEqual(yval, y_tensor.State);
                            CollectionAssert.AreEqual(wval, w_tensor.State);

                            gw_tensor.CheckOverflow();

                            Console.WriteLine($"pass: {inchannels},{outchannels},{batch},{transpose}");

                        }
                    }
                }
            }
        }

        [TestMethod]
        public void SpeedTest() {
            int inchannels = 33, outchannels = 33;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map0D(inchannels));
            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map0D(outchannels));
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels / 3 * 4, outchannels / 3));

            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel0D(inchannels / 3 * 4, outchannels / 3));

            TrivectorKernelProductDense ope = new TrivectorKernelProductDense(inchannels, outchannels);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            ope.Execute(x_tensor, y_tensor, w_tensor, gw_tensor);
            ope.Execute(x_tensor, y_tensor, w_tensor, gw_tensor);
            ope.Execute(x_tensor, y_tensor, w_tensor, gw_tensor);
            ope.Execute(x_tensor, y_tensor, w_tensor, gw_tensor);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds / 4} msec");
        }

        public static Quaternion.QuaternionFilter0D Reference(TrivectorMap0D x, TrivectorMap0D gy, Quaternion.QuaternionFilter0D w) {
            int inchannels = x.Channels, outchannels = gy.Channels, batch = x.Batch;

            Quaternion.QuaternionFilter0D gw = new Quaternion.QuaternionFilter0D(inchannels, outchannels);

            for (int inch, outch = 0; outch < outchannels; outch++) {
                for (inch = 0; inch < inchannels; inch++) {
                    Quaternion.Quaternion sum = 0;
                    Quaternion.Quaternion q = w[inch, outch];

                    for (int th = 0; th < batch; th++) {
                        sum += Trivector.MulQGrad(x[inch, th], gy[outch, th], q);
                    }

                    gw[inch, outch] += sum;
                }
            }

            return gw;
        }
    }
}
