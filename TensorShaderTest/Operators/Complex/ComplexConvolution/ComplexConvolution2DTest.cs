using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.ComplexConvolution;

namespace TensorShaderTest.Operators.Complex {
    [TestClass]
    public class ComplexConvolution2DTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2, 3 }) {
                foreach(int inchannels in new int[]{ 2, 4, 10, 20 }) {
                    foreach(int outchannels in new int[]{ 6, 14 }) {
                        foreach (int kheight in new int[] { 1, 3, 5 }) {
                            foreach (int kwidth in new int[] { 1, 3, 5 }) {
                                foreach (int stride in new int[] { 1, 2, 3 }) {
                                    foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                                        foreach (int inheight in new int[] { 8, 9, 19, 23 }) {
                                            int outwidth = (inwidth - kwidth) / stride + 1, outheight = (inheight - kheight) / stride + 1;

                                            float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                            float[] wval = (new float[kwidth * kheight * inchannels * outchannels / 2]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                            System.Numerics.Complex[] xcval = (new System.Numerics.Complex[xval.Length / 2])
                                                .Select((_, idx) => new System.Numerics.Complex(xval[idx * 2], xval[idx*2 + 1])).ToArray();

                                            System.Numerics.Complex[] wcval = (new System.Numerics.Complex[wval.Length / 2])
                                                .Select((_, idx) => new System.Numerics.Complex(wval[idx * 2], wval[idx * 2 + 1])).ToArray();

                                            ComplexMap2D x = new ComplexMap2D(inchannels / 2, inwidth, inheight, batch, xcval);
                                            ComplexFilter2D w = new ComplexFilter2D(inchannels / 2, outchannels / 2, kwidth, kheight, wcval);

                                            ComplexMap2D y = Reference(x, w, kwidth, kheight, stride);

                                            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
                                            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel2D(inchannels, outchannels / 2, kwidth, kheight), wval);

                                            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map2D(outchannels, outwidth, outheight, batch));

                                            ComplexConvolution2D ope = new ComplexConvolution2D(inwidth, inheight, inchannels, outchannels, kwidth, kheight, stride, gradmode:false, batch);

                                            ope.Execute(x_tensor, w_tensor, y_tensor);

                                            float[] y_expect = y.ToArray();
                                            float[] y_actual = y_tensor.State;

                                            CollectionAssert.AreEqual(xval, x_tensor.State);
                                            CollectionAssert.AreEqual(wval, w_tensor.State);

                                            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{stride},{inwidth},{inheight},{batch}");

                                            Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{stride},{inwidth},{inheight},{batch}");
                                        }
                                    }
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
            foreach (bool gradmode in new bool[] { false, true }) {
                foreach (int batch in new int[] { 1, 2, 3 }) {
                    foreach(int inchannels in new int[]{ 2, 4, 10, 20 }) {
                        foreach(int outchannels in new int[]{ 6, 14 }) {
                            foreach (int kheight in new int[] { 1, 3, 5 }) {
                                foreach (int kwidth in new int[] { 1, 3, 5 }) {
                                    foreach (int stride in new int[] { 1, 2, 3 }) {
                                        foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                                            foreach (int inheight in new int[] { 8, 9, 19, 23 }) {
                                                int outwidth = (inwidth - kwidth) / stride + 1, outheight = (inheight - kheight) / stride + 1;

                                                float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                                float[] wval = (new float[kwidth * kheight * inchannels * outchannels / 2]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                                OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
                                                OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel2D(inchannels, outchannels / 2, kwidth, kheight), wval);

                                                OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map2D(outchannels, outwidth, outheight, batch));

                                                ComplexConvolution2D ope = new ComplexConvolution2D(inwidth, inheight, inchannels, outchannels, kwidth, kheight, stride, gradmode, batch);

                                                ope.Execute(x_tensor, w_tensor, y_tensor);

                                                CollectionAssert.AreEqual(xval, x_tensor.State);
                                                CollectionAssert.AreEqual(wval, w_tensor.State);

                                                y_tensor.CheckOverflow();

                                                Console.WriteLine($"pass: {inchannels},{outchannels},{kwidth},{kheight},{stride},{inwidth},{inheight},{batch},{gradmode}");
                                            }
                                        }
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
            int inwidth = 512, inheight = 512, inchannels = 32, outchannels = 32, ksize = 3, stride = 2;
            int outwidth = (inwidth - ksize) / stride + 1, outheight = (inheight - ksize) / stride + 1;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(inchannels, inwidth, inheight));
            OverflowCheckedTensor w_tensor = new OverflowCheckedTensor(Shape.Kernel2D(inchannels, outchannels / 2, ksize, ksize));

            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map2D(outchannels, outwidth, outheight));

            ComplexConvolution2D ope = new ComplexConvolution2D(inwidth, inheight, inchannels, outchannels, ksize, ksize, stride);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            ope.Execute(x_tensor, w_tensor, y_tensor);
            ope.Execute(x_tensor, w_tensor, y_tensor);
            ope.Execute(x_tensor, w_tensor, y_tensor);
            ope.Execute(x_tensor, w_tensor, y_tensor);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds / 4} msec");
        }

        public static ComplexMap2D Reference(ComplexMap2D x, ComplexFilter2D w, int kwidth, int kheight, int stride) {
            int inchannels = x.Channels, outchannels = w.OutChannels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, outw = (inw - kwidth) / stride + 1, outh = (inh - kheight) / stride + 1;

            ComplexMap2D y = new ComplexMap2D(outchannels, outw, outh, batch);

            for (int kx, ky = 0; ky < kheight; ky++) {
                for (kx = 0; kx < kwidth; kx++) {
                    for (int th = 0; th < batch; th++) {
                        for (int ox, oy = 0; oy < outh; oy++) {
                            for (ox = 0; ox < outw; ox++) {
                                for (int outch = 0; outch < outchannels; outch++) {
                                    System.Numerics.Complex sum = y[outch, ox, oy, th];

                                    for (int inch = 0; inch < inchannels; inch++) {
                                        sum += x[inch, kx + ox * stride, ky + oy * stride, th] * w[inch, outch, kx, ky];
                                    }

                                    y[outch, ox, oy, th] = sum;
                                }
                            }
                        }
                    }
                }
            }

            return y;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 6, outchannels = 8, kwidth = 3, kheight = 5, stride = 2, inwidth = 13, inheight = 17, batch = 3;
            int outwidth = (inwidth - kwidth) / stride + 1, outheight = (inheight - kheight) / stride + 1;

            float[] xval = (new float[batch * inwidth * inheight * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * kheight * outchannels * inchannels / 2]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            System.Numerics.Complex[] xcval = (new System.Numerics.Complex[xval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(xval[idx * 2], xval[idx * 2 + 1])).ToArray();

            System.Numerics.Complex[] wcval = (new System.Numerics.Complex[wval.Length / 2])
                .Select((_, idx) => new System.Numerics.Complex(wval[idx * 2], wval[idx * 2 + 1])).ToArray();

            ComplexMap2D x = new ComplexMap2D(inchannels / 2, inwidth, inheight, batch, xcval);
            ComplexFilter2D w = new ComplexFilter2D(inchannels / 2, outchannels / 2, kwidth, kheight, wcval);

            ComplexMap2D y = Reference(x, w, kwidth, kheight, stride);

            float[] y_expect = {
                -1.080000000e-03f,  1.771005000e+00f,  -8.100000000e-04f,  1.682175000e+00f,  -5.400000000e-04f,  1.593345000e+00f,
                -2.700000000e-04f,  1.504515000e+00f,  -5.400000000e-04f,  1.974585000e+00f,  -2.700000000e-04f,  1.879275000e+00f,
                1.110223025e-16f,  1.783965000e+00f,  2.700000000e-04f,  1.688655000e+00f,  -6.661338148e-16f,  2.178165000e+00f,
                2.700000000e-04f,  2.076375000e+00f,  5.400000000e-04f,  1.974585000e+00f,  8.100000000e-04f,  1.872795000e+00f,
                5.400000000e-04f,  2.381745000e+00f,  8.100000000e-04f,  2.273475000e+00f,  1.080000000e-03f,  2.165205000e+00f,
                1.350000000e-03f,  2.056935000e+00f,  1.080000000e-03f,  2.585325000e+00f,  1.350000000e-03f,  2.470575000e+00f,
                1.620000000e-03f,  2.355825000e+00f,  1.890000000e-03f,  2.241075000e+00f,  1.620000000e-03f,  2.788905000e+00f,
                1.890000000e-03f,  2.667675000e+00f,  2.160000000e-03f,  2.546445000e+00f,  2.430000000e-03f,  2.425215000e+00f,
                5.940000000e-03f,  4.417545000e+00f,  6.210000000e-03f,  4.244475000e+00f,  6.480000000e-03f,  4.071405000e+00f,
                6.750000000e-03f,  3.898335000e+00f,  6.480000000e-03f,  4.621125000e+00f,  6.750000000e-03f,  4.441575000e+00f,
                7.020000000e-03f,  4.262025000e+00f,  7.290000000e-03f,  4.082475000e+00f,  7.020000000e-03f,  4.824705000e+00f,
                7.290000000e-03f,  4.638675000e+00f,  7.560000000e-03f,  4.452645000e+00f,  7.830000000e-03f,  4.266615000e+00f,
                7.560000000e-03f,  5.028285000e+00f,  7.830000000e-03f,  4.835775000e+00f,  8.100000000e-03f,  4.643265000e+00f,
                8.370000000e-03f,  4.450755000e+00f,  8.100000000e-03f,  5.231865000e+00f,  8.370000000e-03f,  5.032875000e+00f,
                8.640000000e-03f,  4.833885000e+00f,  8.910000000e-03f,  4.634895000e+00f,  8.640000000e-03f,  5.435445000e+00f,
                8.910000000e-03f,  5.229975000e+00f,  9.180000000e-03f,  5.024505000e+00f,  9.450000000e-03f,  4.819035000e+00f,
                1.296000000e-02f,  7.064085000e+00f,  1.323000000e-02f,  6.806775000e+00f,  1.350000000e-02f,  6.549465000e+00f,
                1.377000000e-02f,  6.292155000e+00f,  1.350000000e-02f,  7.267665000e+00f,  1.377000000e-02f,  7.003875000e+00f,
                1.404000000e-02f,  6.740085000e+00f,  1.431000000e-02f,  6.476295000e+00f,  1.404000000e-02f,  7.471245000e+00f,
                1.431000000e-02f,  7.200975000e+00f,  1.458000000e-02f,  6.930705000e+00f,  1.485000000e-02f,  6.660435000e+00f,
                1.458000000e-02f,  7.674825000e+00f,  1.485000000e-02f,  7.398075000e+00f,  1.512000000e-02f,  7.121325000e+00f,
                1.539000000e-02f,  6.844575000e+00f,  1.512000000e-02f,  7.878405000e+00f,  1.539000000e-02f,  7.595175000e+00f,
                1.566000000e-02f,  7.311945000e+00f,  1.593000000e-02f,  7.028715000e+00f,  1.566000000e-02f,  8.081985000e+00f,
                1.593000000e-02f,  7.792275000e+00f,  1.620000000e-02f,  7.502565000e+00f,  1.647000000e-02f,  7.212855000e+00f,
                1.998000000e-02f,  9.710625000e+00f,  2.025000000e-02f,  9.369075000e+00f,  2.052000000e-02f,  9.027525000e+00f,
                2.079000000e-02f,  8.685975000e+00f,  2.052000000e-02f,  9.914205000e+00f,  2.079000000e-02f,  9.566175000e+00f,
                2.106000000e-02f,  9.218145000e+00f,  2.133000000e-02f,  8.870115000e+00f,  2.106000000e-02f,  1.011778500e+01f,
                2.133000000e-02f,  9.763275000e+00f,  2.160000000e-02f,  9.408765000e+00f,  2.187000000e-02f,  9.054255000e+00f,
                2.160000000e-02f,  1.032136500e+01f,  2.187000000e-02f,  9.960375000e+00f,  2.214000000e-02f,  9.599385000e+00f,
                2.241000000e-02f,  9.238395000e+00f,  2.214000000e-02f,  1.052494500e+01f,  2.241000000e-02f,  1.015747500e+01f,
                2.268000000e-02f,  9.790005000e+00f,  2.295000000e-02f,  9.422535000e+00f,  2.268000000e-02f,  1.072852500e+01f,
                2.295000000e-02f,  1.035457500e+01f,  2.322000000e-02f,  9.980625000e+00f,  2.349000000e-02f,  9.606675000e+00f,
                2.700000000e-02f,  1.235716500e+01f,  2.727000000e-02f,  1.193137500e+01f,  2.754000000e-02f,  1.150558500e+01f,
                2.781000000e-02f,  1.107979500e+01f,  2.754000000e-02f,  1.256074500e+01f,  2.781000000e-02f,  1.212847500e+01f,
                2.808000000e-02f,  1.169620500e+01f,  2.835000000e-02f,  1.126393500e+01f,  2.808000000e-02f,  1.276432500e+01f,
                2.835000000e-02f,  1.232557500e+01f,  2.862000000e-02f,  1.188682500e+01f,  2.889000000e-02f,  1.144807500e+01f,
                2.862000000e-02f,  1.296790500e+01f,  2.889000000e-02f,  1.252267500e+01f,  2.916000000e-02f,  1.207744500e+01f,
                2.943000000e-02f,  1.163221500e+01f,  2.916000000e-02f,  1.317148500e+01f,  2.943000000e-02f,  1.271977500e+01f,
                2.970000000e-02f,  1.226806500e+01f,  2.997000000e-02f,  1.181635500e+01f,  2.970000000e-02f,  1.337506500e+01f,
                2.997000000e-02f,  1.291687500e+01f,  3.024000000e-02f,  1.245868500e+01f,  3.051000000e-02f,  1.200049500e+01f,
                3.402000000e-02f,  1.500370500e+01f,  3.429000000e-02f,  1.449367500e+01f,  3.456000000e-02f,  1.398364500e+01f,
                3.483000000e-02f,  1.347361500e+01f,  3.456000000e-02f,  1.520728500e+01f,  3.483000000e-02f,  1.469077500e+01f,
                3.510000000e-02f,  1.417426500e+01f,  3.537000000e-02f,  1.365775500e+01f,  3.510000000e-02f,  1.541086500e+01f,
                3.537000000e-02f,  1.488787500e+01f,  3.564000000e-02f,  1.436488500e+01f,  3.591000000e-02f,  1.384189500e+01f,
                3.564000000e-02f,  1.561444500e+01f,  3.591000000e-02f,  1.508497500e+01f,  3.618000000e-02f,  1.455550500e+01f,
                3.645000000e-02f,  1.402603500e+01f,  3.618000000e-02f,  1.581802500e+01f,  3.645000000e-02f,  1.528207500e+01f,
                3.672000000e-02f,  1.474612500e+01f,  3.699000000e-02f,  1.421017500e+01f,  3.672000000e-02f,  1.602160500e+01f,
                3.699000000e-02f,  1.547917500e+01f,  3.726000000e-02f,  1.493674500e+01f,  3.753000000e-02f,  1.439431500e+01f,
                4.104000000e-02f,  1.765024500e+01f,  4.131000000e-02f,  1.705597500e+01f,  4.158000000e-02f,  1.646170500e+01f,
                4.185000000e-02f,  1.586743500e+01f,  4.158000000e-02f,  1.785382500e+01f,  4.185000000e-02f,  1.725307500e+01f,
                4.212000000e-02f,  1.665232500e+01f,  4.239000000e-02f,  1.605157500e+01f,  4.212000000e-02f,  1.805740500e+01f,
                4.239000000e-02f,  1.745017500e+01f,  4.266000000e-02f,  1.684294500e+01f,  4.293000000e-02f,  1.623571500e+01f,
                4.266000000e-02f,  1.826098500e+01f,  4.293000000e-02f,  1.764727500e+01f,  4.320000000e-02f,  1.703356500e+01f,
                4.347000000e-02f,  1.641985500e+01f,  4.320000000e-02f,  1.846456500e+01f,  4.347000000e-02f,  1.784437500e+01f,
                4.374000000e-02f,  1.722418500e+01f,  4.401000000e-02f,  1.660399500e+01f,  4.374000000e-02f,  1.866814500e+01f,
                4.401000000e-02f,  1.804147500e+01f,  4.428000000e-02f,  1.741480500e+01f,  4.455000000e-02f,  1.678813500e+01f,
                5.859000000e-02f,  2.426659500e+01f,  5.886000000e-02f,  2.346172500e+01f,  5.913000000e-02f,  2.265685500e+01f,
                5.940000000e-02f,  2.185198500e+01f,  5.913000000e-02f,  2.447017500e+01f,  5.940000000e-02f,  2.365882500e+01f,
                5.967000000e-02f,  2.284747500e+01f,  5.994000000e-02f,  2.203612500e+01f,  5.967000000e-02f,  2.467375500e+01f,
                5.994000000e-02f,  2.385592500e+01f,  6.021000000e-02f,  2.303809500e+01f,  6.048000000e-02f,  2.222026500e+01f,
                6.021000000e-02f,  2.487733500e+01f,  6.048000000e-02f,  2.405302500e+01f,  6.075000000e-02f,  2.322871500e+01f,
                6.102000000e-02f,  2.240440500e+01f,  6.075000000e-02f,  2.508091500e+01f,  6.102000000e-02f,  2.425012500e+01f,
                6.129000000e-02f,  2.341933500e+01f,  6.156000000e-02f,  2.258854500e+01f,  6.129000000e-02f,  2.528449500e+01f,
                6.156000000e-02f,  2.444722500e+01f,  6.183000000e-02f,  2.360995500e+01f,  6.210000000e-02f,  2.277268500e+01f,
                6.561000000e-02f,  2.691313500e+01f,  6.588000000e-02f,  2.602402500e+01f,  6.615000000e-02f,  2.513491500e+01f,
                6.642000000e-02f,  2.424580500e+01f,  6.615000000e-02f,  2.711671500e+01f,  6.642000000e-02f,  2.622112500e+01f,
                6.669000000e-02f,  2.532553500e+01f,  6.696000000e-02f,  2.442994500e+01f,  6.669000000e-02f,  2.732029500e+01f,
                6.696000000e-02f,  2.641822500e+01f,  6.723000000e-02f,  2.551615500e+01f,  6.750000000e-02f,  2.461408500e+01f,
                6.723000000e-02f,  2.752387500e+01f,  6.750000000e-02f,  2.661532500e+01f,  6.777000000e-02f,  2.570677500e+01f,
                6.804000000e-02f,  2.479822500e+01f,  6.777000000e-02f,  2.772745500e+01f,  6.804000000e-02f,  2.681242500e+01f,
                6.831000000e-02f,  2.589739500e+01f,  6.858000000e-02f,  2.498236500e+01f,  6.831000000e-02f,  2.793103500e+01f,
                6.858000000e-02f,  2.700952500e+01f,  6.885000000e-02f,  2.608801500e+01f,  6.912000000e-02f,  2.516650500e+01f,
                7.263000000e-02f,  2.955967500e+01f,  7.290000000e-02f,  2.858632500e+01f,  7.317000000e-02f,  2.761297500e+01f,
                7.344000000e-02f,  2.663962500e+01f,  7.317000000e-02f,  2.976325500e+01f,  7.344000000e-02f,  2.878342500e+01f,
                7.371000000e-02f,  2.780359500e+01f,  7.398000000e-02f,  2.682376500e+01f,  7.371000000e-02f,  2.996683500e+01f,
                7.398000000e-02f,  2.898052500e+01f,  7.425000000e-02f,  2.799421500e+01f,  7.452000000e-02f,  2.700790500e+01f,
                7.425000000e-02f,  3.017041500e+01f,  7.452000000e-02f,  2.917762500e+01f,  7.479000000e-02f,  2.818483500e+01f,
                7.506000000e-02f,  2.719204500e+01f,  7.479000000e-02f,  3.037399500e+01f,  7.506000000e-02f,  2.937472500e+01f,
                7.533000000e-02f,  2.837545500e+01f,  7.560000000e-02f,  2.737618500e+01f,  7.533000000e-02f,  3.057757500e+01f,
                7.560000000e-02f,  2.957182500e+01f,  7.587000000e-02f,  2.856607500e+01f,  7.614000000e-02f,  2.756032500e+01f,
                7.965000000e-02f,  3.220621500e+01f,  7.992000000e-02f,  3.114862500e+01f,  8.019000000e-02f,  3.009103500e+01f,
                8.046000000e-02f,  2.903344500e+01f,  8.019000000e-02f,  3.240979500e+01f,  8.046000000e-02f,  3.134572500e+01f,
                8.073000000e-02f,  3.028165500e+01f,  8.100000000e-02f,  2.921758500e+01f,  8.073000000e-02f,  3.261337500e+01f,
                8.100000000e-02f,  3.154282500e+01f,  8.127000000e-02f,  3.047227500e+01f,  8.154000000e-02f,  2.940172500e+01f,
                8.127000000e-02f,  3.281695500e+01f,  8.154000000e-02f,  3.173992500e+01f,  8.181000000e-02f,  3.066289500e+01f,
                8.208000000e-02f,  2.958586500e+01f,  8.181000000e-02f,  3.302053500e+01f,  8.208000000e-02f,  3.193702500e+01f,
                8.235000000e-02f,  3.085351500e+01f,  8.262000000e-02f,  2.977000500e+01f,  8.235000000e-02f,  3.322411500e+01f,
                8.262000000e-02f,  3.213412500e+01f,  8.289000000e-02f,  3.104413500e+01f,  8.316000000e-02f,  2.995414500e+01f,
                8.667000000e-02f,  3.485275500e+01f,  8.694000000e-02f,  3.371092500e+01f,  8.721000000e-02f,  3.256909500e+01f,
                8.748000000e-02f,  3.142726500e+01f,  8.721000000e-02f,  3.505633500e+01f,  8.748000000e-02f,  3.390802500e+01f,
                8.775000000e-02f,  3.275971500e+01f,  8.802000000e-02f,  3.161140500e+01f,  8.775000000e-02f,  3.525991500e+01f,
                8.802000000e-02f,  3.410512500e+01f,  8.829000000e-02f,  3.295033500e+01f,  8.856000000e-02f,  3.179554500e+01f,
                8.829000000e-02f,  3.546349500e+01f,  8.856000000e-02f,  3.430222500e+01f,  8.883000000e-02f,  3.314095500e+01f,
                8.910000000e-02f,  3.197968500e+01f,  8.883000000e-02f,  3.566707500e+01f,  8.910000000e-02f,  3.449932500e+01f,
                8.937000000e-02f,  3.333157500e+01f,  8.964000000e-02f,  3.216382500e+01f,  8.937000000e-02f,  3.587065500e+01f,
                8.964000000e-02f,  3.469642500e+01f,  8.991000000e-02f,  3.352219500e+01f,  9.018000000e-02f,  3.234796500e+01f,
                9.369000000e-02f,  3.749929500e+01f,  9.396000000e-02f,  3.627322500e+01f,  9.423000000e-02f,  3.504715500e+01f,
                9.450000000e-02f,  3.382108500e+01f,  9.423000000e-02f,  3.770287500e+01f,  9.450000000e-02f,  3.647032500e+01f,
                9.477000000e-02f,  3.523777500e+01f,  9.504000000e-02f,  3.400522500e+01f,  9.477000000e-02f,  3.790645500e+01f,
                9.504000000e-02f,  3.666742500e+01f,  9.531000000e-02f,  3.542839500e+01f,  9.558000000e-02f,  3.418936500e+01f,
                9.531000000e-02f,  3.811003500e+01f,  9.558000000e-02f,  3.686452500e+01f,  9.585000000e-02f,  3.561901500e+01f,
                9.612000000e-02f,  3.437350500e+01f,  9.585000000e-02f,  3.831361500e+01f,  9.612000000e-02f,  3.706162500e+01f,
                9.639000000e-02f,  3.580963500e+01f,  9.666000000e-02f,  3.455764500e+01f,  9.639000000e-02f,  3.851719500e+01f,
                9.666000000e-02f,  3.725872500e+01f,  9.693000000e-02f,  3.600025500e+01f,  9.720000000e-02f,  3.474178500e+01f,
                1.007100000e-01f,  4.014583500e+01f,  1.009800000e-01f,  3.883552500e+01f,  1.012500000e-01f,  3.752521500e+01f,
                1.015200000e-01f,  3.621490500e+01f,  1.012500000e-01f,  4.034941500e+01f,  1.015200000e-01f,  3.903262500e+01f,
                1.017900000e-01f,  3.771583500e+01f,  1.020600000e-01f,  3.639904500e+01f,  1.017900000e-01f,  4.055299500e+01f,
                1.020600000e-01f,  3.922972500e+01f,  1.023300000e-01f,  3.790645500e+01f,  1.026000000e-01f,  3.658318500e+01f,
                1.023300000e-01f,  4.075657500e+01f,  1.026000000e-01f,  3.942682500e+01f,  1.028700000e-01f,  3.809707500e+01f,
                1.031400000e-01f,  3.676732500e+01f,  1.028700000e-01f,  4.096015500e+01f,  1.031400000e-01f,  3.962392500e+01f,
                1.034100000e-01f,  3.828769500e+01f,  1.036800000e-01f,  3.695146500e+01f,  1.034100000e-01f,  4.116373500e+01f,
                1.036800000e-01f,  3.982102500e+01f,  1.039500000e-01f,  3.847831500e+01f,  1.042200000e-01f,  3.713560500e+01f,
                1.182600000e-01f,  4.676218500e+01f,  1.185300000e-01f,  4.524127500e+01f,  1.188000000e-01f,  4.372036500e+01f,
                1.190700000e-01f,  4.219945500e+01f,  1.188000000e-01f,  4.696576500e+01f,  1.190700000e-01f,  4.543837500e+01f,
                1.193400000e-01f,  4.391098500e+01f,  1.196100000e-01f,  4.238359500e+01f,  1.193400000e-01f,  4.716934500e+01f,
                1.196100000e-01f,  4.563547500e+01f,  1.198800000e-01f,  4.410160500e+01f,  1.201500000e-01f,  4.256773500e+01f,
                1.198800000e-01f,  4.737292500e+01f,  1.201500000e-01f,  4.583257500e+01f,  1.204200000e-01f,  4.429222500e+01f,
                1.206900000e-01f,  4.275187500e+01f,  1.204200000e-01f,  4.757650500e+01f,  1.206900000e-01f,  4.602967500e+01f,
                1.209600000e-01f,  4.448284500e+01f,  1.212300000e-01f,  4.293601500e+01f,  1.209600000e-01f,  4.778008500e+01f,
                1.212300000e-01f,  4.622677500e+01f,  1.215000000e-01f,  4.467346500e+01f,  1.217700000e-01f,  4.312015500e+01f,
                1.252800000e-01f,  4.940872500e+01f,  1.255500000e-01f,  4.780357500e+01f,  1.258200000e-01f,  4.619842500e+01f,
                1.260900000e-01f,  4.459327500e+01f,  1.258200000e-01f,  4.961230500e+01f,  1.260900000e-01f,  4.800067500e+01f,
                1.263600000e-01f,  4.638904500e+01f,  1.266300000e-01f,  4.477741500e+01f,  1.263600000e-01f,  4.981588500e+01f,
                1.266300000e-01f,  4.819777500e+01f,  1.269000000e-01f,  4.657966500e+01f,  1.271700000e-01f,  4.496155500e+01f,
                1.269000000e-01f,  5.001946500e+01f,  1.271700000e-01f,  4.839487500e+01f,  1.274400000e-01f,  4.677028500e+01f,
                1.277100000e-01f,  4.514569500e+01f,  1.274400000e-01f,  5.022304500e+01f,  1.277100000e-01f,  4.859197500e+01f,
                1.279800000e-01f,  4.696090500e+01f,  1.282500000e-01f,  4.532983500e+01f,  1.279800000e-01f,  5.042662500e+01f,
                1.282500000e-01f,  4.878907500e+01f,  1.285200000e-01f,  4.715152500e+01f,  1.287900000e-01f,  4.551397500e+01f,
                1.323000000e-01f,  5.205526500e+01f,  1.325700000e-01f,  5.036587500e+01f,  1.328400000e-01f,  4.867648500e+01f,
                1.331100000e-01f,  4.698709500e+01f,  1.328400000e-01f,  5.225884500e+01f,  1.331100000e-01f,  5.056297500e+01f,
                1.333800000e-01f,  4.886710500e+01f,  1.336500000e-01f,  4.717123500e+01f,  1.333800000e-01f,  5.246242500e+01f,
                1.336500000e-01f,  5.076007500e+01f,  1.339200000e-01f,  4.905772500e+01f,  1.341900000e-01f,  4.735537500e+01f,
                1.339200000e-01f,  5.266600500e+01f,  1.341900000e-01f,  5.095717500e+01f,  1.344600000e-01f,  4.924834500e+01f,
                1.347300000e-01f,  4.753951500e+01f,  1.344600000e-01f,  5.286958500e+01f,  1.347300000e-01f,  5.115427500e+01f,
                1.350000000e-01f,  4.943896500e+01f,  1.352700000e-01f,  4.772365500e+01f,  1.350000000e-01f,  5.307316500e+01f,
                1.352700000e-01f,  5.135137500e+01f,  1.355400000e-01f,  4.962958500e+01f,  1.358100000e-01f,  4.790779500e+01f,
                1.393200000e-01f,  5.470180500e+01f,  1.395900000e-01f,  5.292817500e+01f,  1.398600000e-01f,  5.115454500e+01f,
                1.401300000e-01f,  4.938091500e+01f,  1.398600000e-01f,  5.490538500e+01f,  1.401300000e-01f,  5.312527500e+01f,
                1.404000000e-01f,  5.134516500e+01f,  1.406700000e-01f,  4.956505500e+01f,  1.404000000e-01f,  5.510896500e+01f,
                1.406700000e-01f,  5.332237500e+01f,  1.409400000e-01f,  5.153578500e+01f,  1.412100000e-01f,  4.974919500e+01f,
                1.409400000e-01f,  5.531254500e+01f,  1.412100000e-01f,  5.351947500e+01f,  1.414800000e-01f,  5.172640500e+01f,
                1.417500000e-01f,  4.993333500e+01f,  1.414800000e-01f,  5.551612500e+01f,  1.417500000e-01f,  5.371657500e+01f,
                1.420200000e-01f,  5.191702500e+01f,  1.422900000e-01f,  5.011747500e+01f,  1.420200000e-01f,  5.571970500e+01f,
                1.422900000e-01f,  5.391367500e+01f,  1.425600000e-01f,  5.210764500e+01f,  1.428300000e-01f,  5.030161500e+01f,
                1.463400000e-01f,  5.734834500e+01f,  1.466100000e-01f,  5.549047500e+01f,  1.468800000e-01f,  5.363260500e+01f,
                1.471500000e-01f,  5.177473500e+01f,  1.468800000e-01f,  5.755192500e+01f,  1.471500000e-01f,  5.568757500e+01f,
                1.474200000e-01f,  5.382322500e+01f,  1.476900000e-01f,  5.195887500e+01f,  1.474200000e-01f,  5.775550500e+01f,
                1.476900000e-01f,  5.588467500e+01f,  1.479600000e-01f,  5.401384500e+01f,  1.482300000e-01f,  5.214301500e+01f,
                1.479600000e-01f,  5.795908500e+01f,  1.482300000e-01f,  5.608177500e+01f,  1.485000000e-01f,  5.420446500e+01f,
                1.487700000e-01f,  5.232715500e+01f,  1.485000000e-01f,  5.816266500e+01f,  1.487700000e-01f,  5.627887500e+01f,
                1.490400000e-01f,  5.439508500e+01f,  1.493100000e-01f,  5.251129500e+01f,  1.490400000e-01f,  5.836624500e+01f,
                1.493100000e-01f,  5.647597500e+01f,  1.495800000e-01f,  5.458570500e+01f,  1.498500000e-01f,  5.269543500e+01f,
                1.533600000e-01f,  5.999488500e+01f,  1.536300000e-01f,  5.805277500e+01f,  1.539000000e-01f,  5.611066500e+01f,
                1.541700000e-01f,  5.416855500e+01f,  1.539000000e-01f,  6.019846500e+01f,  1.541700000e-01f,  5.824987500e+01f,
                1.544400000e-01f,  5.630128500e+01f,  1.547100000e-01f,  5.435269500e+01f,  1.544400000e-01f,  6.040204500e+01f,
                1.547100000e-01f,  5.844697500e+01f,  1.549800000e-01f,  5.649190500e+01f,  1.552500000e-01f,  5.453683500e+01f,
                1.549800000e-01f,  6.060562500e+01f,  1.552500000e-01f,  5.864407500e+01f,  1.555200000e-01f,  5.668252500e+01f,
                1.557900000e-01f,  5.472097500e+01f,  1.555200000e-01f,  6.080920500e+01f,  1.557900000e-01f,  5.884117500e+01f,
                1.560600000e-01f,  5.687314500e+01f,  1.563300000e-01f,  5.490511500e+01f,  1.560600000e-01f,  6.101278500e+01f,
                1.563300000e-01f,  5.903827500e+01f,  1.566000000e-01f,  5.706376500e+01f,  1.568700000e-01f,  5.508925500e+01f,
                1.603800000e-01f,  6.264142500e+01f,  1.606500000e-01f,  6.061507500e+01f,  1.609200000e-01f,  5.858872500e+01f,
                1.611900000e-01f,  5.656237500e+01f,  1.609200000e-01f,  6.284500500e+01f,  1.611900000e-01f,  6.081217500e+01f,
                1.614600000e-01f,  5.877934500e+01f,  1.617300000e-01f,  5.674651500e+01f,  1.614600000e-01f,  6.304858500e+01f,
                1.617300000e-01f,  6.100927500e+01f,  1.620000000e-01f,  5.896996500e+01f,  1.622700000e-01f,  5.693065500e+01f,
                1.620000000e-01f,  6.325216500e+01f,  1.622700000e-01f,  6.120637500e+01f,  1.625400000e-01f,  5.916058500e+01f,
                1.628100000e-01f,  5.711479500e+01f,  1.625400000e-01f,  6.345574500e+01f,  1.628100000e-01f,  6.140347500e+01f,
                1.630800000e-01f,  5.935120500e+01f,  1.633500000e-01f,  5.729893500e+01f,  1.630800000e-01f,  6.365932500e+01f,
                1.633500000e-01f,  6.160057500e+01f,  1.636200000e-01f,  5.954182500e+01f,  1.638900000e-01f,  5.748307500e+01f,
            };

            float[] y_actual = y.ToArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{stride},{inwidth},{inheight},{batch}");
        }
    }
}
