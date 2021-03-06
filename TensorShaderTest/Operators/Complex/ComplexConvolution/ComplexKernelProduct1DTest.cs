using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.ComplexConvolution;

namespace TensorShaderTest.Operators.Complex {
    [TestClass]
    public class ComplexKernelProduct1DTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2, 3 }) {
                foreach (int inchannels in new int[] { 2, 4, 10, 20 }) {
                    foreach (int outchannels in new int[] { 6, 14 }) {
                        foreach (int kwidth in new int[] { 1, 3, 5 }) {
                            foreach (int stride in new int[] { 1, 2, 3 }) {
                                foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                                    int outwidth = (inwidth - kwidth) / stride + 1;

                                    float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                    float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                    System.Numerics.Complex[] xcval = (new System.Numerics.Complex[xval.Length / 2])
                                        .Select((_, idx) => new System.Numerics.Complex(xval[idx * 2], xval[idx * 2 + 1])).ToArray();

                                    System.Numerics.Complex[] ycval = (new System.Numerics.Complex[yval.Length / 2])
                                        .Select((_, idx) => new System.Numerics.Complex(yval[idx * 2], yval[idx * 2 + 1])).ToArray();

                                    ComplexMap1D x = new ComplexMap1D(inchannels / 2, inwidth, batch, xcval);
                                    ComplexMap1D y = new ComplexMap1D(outchannels / 2, outwidth, batch, ycval);

                                    ComplexFilter1D gw = Reference(x, y, kwidth, stride);

                                    OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth, batch), xval);
                                    OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth, batch), yval);

                                    OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels / 2, kwidth));

                                    ComplexKernelProduct1D ope = new ComplexKernelProduct1D(inwidth, inchannels, outchannels, kwidth, stride, transpose: false, batch);

                                    ope.Execute(x_tensor, y_tensor, gw_tensor);

                                    float[] gw_expect = gw.ToArray();
                                    float[] gw_actual = gw_tensor.State;

                                    CollectionAssert.AreEqual(xval, x_tensor.State);
                                    CollectionAssert.AreEqual(yval, y_tensor.State);

                                    AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{stride},{inwidth},{batch}");

                                    Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{stride},{inwidth},{batch}");

                                }
                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void OverflowTest() {
            foreach(bool transpose in new bool[] { false, true }) { 
                foreach (int batch in new int[] { 1, 2, 3 }) {
                    foreach (int inchannels in new int[] { 2, 4, 10, 20 }) {
                        foreach (int outchannels in new int[] { 6, 14 }) {
                            foreach (int kwidth in new int[] { 1, 3, 5 }) {
                                foreach (int stride in new int[] { 1, 2, 3 }) {
                                    foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                                        int outwidth = (inwidth - kwidth) / stride + 1;

                                        float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                        float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                        OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth, batch), xval);
                                        OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth, batch), yval);

                                        OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels / 2, kwidth));

                                        ComplexKernelProduct1D ope = new ComplexKernelProduct1D(inwidth, inchannels, outchannels, kwidth, stride, transpose, batch);

                                        ope.Execute(x_tensor, y_tensor, gw_tensor);

                                        CollectionAssert.AreEqual(xval, x_tensor.State);
                                        CollectionAssert.AreEqual(yval, y_tensor.State);

                                        gw_tensor.CheckOverflow();

                                        Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{stride},{inwidth},{batch},{transpose}");

                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 512, inchannels = 32, outchannels = 32, ksize = 3, stride = 2;
            int outwidth = (inwidth - ksize) / stride + 1;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map1D(inchannels, inwidth));
            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map1D(outchannels, outwidth));

            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel1D(inchannels, outchannels / 2, ksize));

            ComplexKernelProduct1D ope = new ComplexKernelProduct1D(inwidth, inchannels, outchannels, ksize, stride);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            ope.Execute(x_tensor, y_tensor, gw_tensor);
            ope.Execute(x_tensor, y_tensor, gw_tensor);
            ope.Execute(x_tensor, y_tensor, gw_tensor);
            ope.Execute(x_tensor, y_tensor, gw_tensor);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds / 4} msec");
        }

        public static ComplexFilter1D Reference(ComplexMap1D x, ComplexMap1D gy, int kwidth, int stride) {
            int inchannels = x.Channels, outchannels = gy.Channels, batch = x.Batch;
            int inw = x.Width, outw = gy.Width;

            if (outw != (inw - kwidth) / stride + 1) {
                throw new ArgumentException("mismatch shape");
            }

            ComplexFilter1D w = new ComplexFilter1D(inchannels, outchannels, kwidth);

            Func<System.Numerics.Complex, System.Numerics.Complex, System.Numerics.Complex> mul_grad = (z1, z2) => {
                return new System.Numerics.Complex(z1.Real * z2.Real + z1.Imaginary * z2.Imaginary, z1.Imaginary * z2.Real - z1.Real * z2.Imaginary);
            };

            for (int kx = 0; kx < kwidth; kx++) {
                for (int th = 0; th < batch; th++) {
                    for (int inch, outch = 0; outch < outchannels; outch++) {
                        for (inch = 0; inch < inchannels; inch++) {
                            System.Numerics.Complex sum = 0;

                            for (int ix = kx, ox = 0; ox < outw; ix += stride, ox++) {
                                sum += mul_grad(gy[outch, ox, th], x[inch, ix, th]);
                            }

                            w[inch, outch, kx] += sum;
                        }

                    }
                }
            }

            return w;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 6, outchannels = 8, kwidth = 3, stride = 2, inwidth = 13;
            int outwidth = (inwidth - kwidth) / stride + 1, batch = 1;

            float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            System.Numerics.Complex[] xcval = (new System.Numerics.Complex[xval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(xval[idx * 2], xval[idx * 2 + 1])).ToArray();

            System.Numerics.Complex[] ycval = (new System.Numerics.Complex[yval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(yval[idx * 2], yval[idx * 2 + 1])).ToArray();

            ComplexMap1D x = new ComplexMap1D(inchannels / 2, inwidth, batch, xcval);
            ComplexMap1D y = new ComplexMap1D(outchannels / 2, outwidth, batch, ycval);

            ComplexFilter1D gw = Reference(x, y, kwidth, stride);

            float[] gw_expect = {
                6.336000000e-03f,  -3.420000000e-04f,  6.972000000e-03f,  -3.540000000e-04f,  7.608000000e-03f,  -3.660000000e-04f,
                5.604000000e-03f,  -3.300000000e-04f,  6.192000000e-03f,  -3.420000000e-04f,  6.780000000e-03f,  -3.540000000e-04f,
                4.872000000e-03f,  -3.180000000e-04f,  5.412000000e-03f,  -3.300000000e-04f,  5.952000000e-03f,  -3.420000000e-04f,
                4.140000000e-03f,  -3.060000000e-04f,  4.632000000e-03f,  -3.180000000e-04f,  5.124000000e-03f,  -3.300000000e-04f,
                8.244000000e-03f,  -3.780000000e-04f,  8.880000000e-03f,  -3.900000000e-04f,  9.516000000e-03f,  -4.020000000e-04f,
                7.368000000e-03f,  -3.660000000e-04f,  7.956000000e-03f,  -3.780000000e-04f,  8.544000000e-03f,  -3.900000000e-04f,
                6.492000000e-03f,  -3.540000000e-04f,  7.032000000e-03f,  -3.660000000e-04f,  7.572000000e-03f,  -3.780000000e-04f,
                5.616000000e-03f,  -3.420000000e-04f,  6.108000000e-03f,  -3.540000000e-04f,  6.600000000e-03f,  -3.660000000e-04f,
                1.015200000e-02f,  -4.140000000e-04f,  1.078800000e-02f,  -4.260000000e-04f,  1.142400000e-02f,  -4.380000000e-04f,
                9.132000000e-03f,  -4.020000000e-04f,  9.720000000e-03f,  -4.140000000e-04f,  1.030800000e-02f,  -4.260000000e-04f,
                8.112000000e-03f,  -3.900000000e-04f,  8.652000000e-03f,  -4.020000000e-04f,  9.192000000e-03f,  -4.140000000e-04f,
                7.092000000e-03f,  -3.780000000e-04f,  7.584000000e-03f,  -3.900000000e-04f,  8.076000000e-03f,  -4.020000000e-04f,
            };

            float[] gw_actual = gw.ToArray();

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{stride},{inwidth},{batch}");
        }
    }
}
