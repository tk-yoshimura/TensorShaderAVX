using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection2D;

namespace TensorShaderTest.Operators.Connection2D {
    [TestClass]
    public class SpaceToChannelTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 3, 5 }) {
                    foreach (int scale in new int[] { 2, 3, 4 }) {
                        foreach (int outwidth in new int[] { 5, 7, 11 }) {
                            foreach (int outheight in new int[] { 5, 7, 11 }) {
                                int inwidth = outwidth * scale, inheight = outheight * scale, outchannels = inchannels * scale * scale;

                                float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

                                Map2D x = new Map2D(inchannels, inwidth, inheight, batch, xval);

                                Map2D y = Reference(x, scale);

                                OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
                                OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map2D(outchannels, outwidth, outheight, batch));

                                SpaceToChannel ope = new SpaceToChannel(inwidth, inheight, inchannels, scale, batch);

                                ope.Execute(x_tensor, y_tensor);

                                float[] y_expect = y.ToArray();
                                float[] y_actual = y_tensor.State;

                                CollectionAssert.AreEqual(xval, x_tensor.State);

                                AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{scale},{inwidth},{inheight},{batch}");

                                Console.WriteLine($"pass: {inchannels},{outchannels},{scale},{inwidth},{inheight},{batch}");

                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 512, inheight = 512, inchannels = 32, scale = 2;
            int outwidth = inwidth / scale, outheight = inheight / scale, outchannels = inchannels * (scale * scale);

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(inchannels, inwidth, inheight));
            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map2D(outchannels, outwidth, outheight));

            SpaceToChannel ope = new SpaceToChannel(inwidth, inheight, inchannels, scale);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            ope.Execute(x_tensor, y_tensor);
            ope.Execute(x_tensor, y_tensor);
            ope.Execute(x_tensor, y_tensor);
            ope.Execute(x_tensor, y_tensor);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds / 4} msec");
        }

        public static Map2D Reference(Map2D x, int scale) {
            int inw = x.Width, inh = x.Height, inchannels = x.Channels, batch = x.Batch;
            if (inw % scale != 0 || inh % scale != 0) {
                throw new ArgumentException(nameof(scale));
            }

            int outw = inw / scale, outh = inh / scale;
            int outchannels = inchannels * scale * scale;

            Map2D y = new Map2D(outchannels, outw, outh, batch);

            for (int th = 0; th < batch; th++) {
                for (int ox, oy = 0; oy < outh; oy++) {
                    for (ox = 0; ox < outw; ox++) {
                        for (int kx, ky = 0; ky < scale; ky++) {
                            for (kx = 0; kx < scale; kx++) {
                                for (int inch = 0; inch < inchannels; inch++) {
                                    int outch = inch + kx * inchannels + ky * inchannels * scale;

                                    y[outch, ox, oy, th] = x[inch, ox * scale + kx, oy * scale + ky, th];

                                }
                            }
                        }
                    }
                }

            }

            return y;
        }

        public static Map2D OptimizedReference(Map2D x, int scale) {
            int inw = x.Width, inh = x.Height, inchannels = x.Channels, batch = x.Batch;
            if (inw % scale != 0 || inh % scale != 0) {
                throw new ArgumentException(nameof(scale));
            }

            int outw = inw / scale, outh = inh / scale;
            int outchannels = inchannels * scale * scale;

            Map2D y = new Map2D(outchannels, outw, outh, batch);

            for (int th = 0; th < batch; th++) {
                for (int ox, oy = 0; oy < outh; oy++) {
                    for (ox = 0; ox < outw; ox++) {
                        int inmap_org = (ox * scale + oy * outw * scale * scale) * inchannels + th * inw * inh * inchannels;
                        int outmap_idx = (ox + oy * outw) * outchannels + th * outw * outh * outchannels;

                        for (int ky = 0; ky < scale; ky++) {
                            int inmap_idx = inmap_org;

                            for (int i = 0; i < scale * inchannels; i++) {
                                y[outmap_idx] = x[inmap_idx];

                                inmap_idx++;
                                outmap_idx++;
                            }

                            inmap_org += outw * scale * inchannels;
                        }
                    }
                }

            }

            return y;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 12, scale = 2, inwidth = 7, inheight = 5;

            float[] xval = (new float[inwidth * inheight * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();

            Map2D x = new Map2D(inchannels, inwidth, inheight, 1, xval);

            Map2D y = Reference(ChannelToSpaceTest.Reference(x, scale), scale);

            CollectionAssert.AreEqual(x.ToArray(), y.ToArray());
        }

        [TestMethod]
        public void OptimizeTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int inchannels in new int[] { 3, 5 }) {
                    foreach (int scale in new int[] { 2, 3, 4 }) {
                        foreach (int outwidth in new int[] { 5, 7, 11 }) {
                            foreach (int outheight in new int[] { 5, 7, 11 }) {
                                int inwidth = outwidth * scale, inheight = outheight * scale, outchannels = inchannels * scale * scale;

                                float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

                                Map2D x = new Map2D(inchannels, inwidth, inheight, batch, xval);

                                Map2D y = Reference(x, scale);
                                Map2D y_optimized = OptimizedReference(x, scale);

                                float[] y_expect = y.ToArray();
                                float[] y_actual = y_optimized.ToArray();

                                AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{scale},{inwidth},{inheight},{batch}");

                                Console.WriteLine($"pass: {inchannels},{outchannels},{scale},{inwidth},{inheight},{batch}");

                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }
    }
}
