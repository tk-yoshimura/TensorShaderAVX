using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection3D;

namespace TensorShaderTest.Operators.Connection3D {
    [TestClass]
    public class EdgePaddingTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach (int channels in new int[] { 1, 2, 3, 4, 5, 6, 7, 8 }) {
                    foreach ((int leftpad, int rightpad) in new (int, int)[] { (0, 0), (0, 1), (1, 0), (1, 2), (2, 1) }) {
                        foreach ((int toppad, int bottompad) in new (int, int)[] { (0, 0), (0, 1), (1, 0), (1, 2), (2, 1) }) {
                            foreach ((int frontpad, int rearpad) in new (int, int)[] { (0, 0), (0, 1), (1, 0), (1, 2), (2, 1) }) {
                                foreach ((int inwidth, int inheight, int indepth) in new (int, int, int)[] { (6, 6, 9), (6, 9, 6), (9, 6, 6) }) {
                                    int outwidth = inwidth + leftpad + rightpad, outheight = inheight + toppad + bottompad, outdepth = indepth + frontpad + rearpad;

                                    float[] xval = (new float[inwidth * inheight * indepth * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

                                    Map3D x = new Map3D(channels, inwidth, inheight, indepth, batch, xval);

                                    Map3D y = Reference(x, leftpad, rightpad, toppad, bottompad, frontpad, rearpad);

                                    OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map3D(channels, inwidth, inheight, indepth, batch), xval);
                                    OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map3D(channels, outwidth, outheight, outdepth, batch));

                                    EdgePadding ope = new EdgePadding(inwidth, inheight, indepth, channels, leftpad, rightpad, toppad, bottompad, frontpad, rearpad, batch);

                                    ope.Execute(x_tensor, y_tensor);

                                    float[] y_expect = y.ToArray();
                                    float[] y_actual = y_tensor.State;

                                    CollectionAssert.AreEqual(xval, x_tensor.State);

                                    AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{leftpad},{rightpad},{toppad},{bottompad},{frontpad},{rearpad},{inwidth},{inheight},{batch}");

                                    Console.WriteLine($"pass: {channels},{leftpad},{rightpad},{toppad},{bottompad},{frontpad},{rearpad},{inwidth},{inheight},{batch}");

                                }
                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        public static Map3D Reference(Map3D x, int leftpad, int rightpad, int toppad, int bottompad, int frontpad, int rearpad) {
            int channels = x.Channels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, ind = x.Depth;
            int outw = inw + leftpad + rightpad, outh = inh + toppad + bottompad, outd = ind + frontpad + rearpad;

            Map3D y = new Map3D(channels, outw, outh, outd, batch);

            for (int th = 0; th < batch; th++) {
                for (int ox, oy, oz = 0; oz < outd; oz++) {
                    int iz = Math.Min(ind - 1, Math.Max(0, oz - frontpad));

                    for (oy = 0; oy < outh; oy++) {
                        int iy = Math.Min(inh - 1, Math.Max(0, oy - toppad));

                        for (ox = 0; ox < outw; ox++) {
                            int ix = Math.Min(inw - 1, Math.Max(0, ox - leftpad));

                            for (int ch = 0; ch < channels; ch++) {
                                y[ch, ox, oy, oz, th] = x[ch, ix, iy, iz, th];
                            }
                        }
                    }
                }
            }

            return y;
        }

        [TestMethod]
        public void SpeedTest() {
            int inwidth = 64, inheight = 64, indepth = 64, channels = 32, leftpad = 1, rightpad = 1, toppad = 1, bottompad = 1, frontpad = 1, rearpad = 1;
            int outwidth = inwidth + leftpad + rightpad, outheight = inheight + toppad + bottompad, outdepth = indepth + frontpad + rearpad;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map3D(channels, inwidth, inheight, indepth));
            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map3D(channels, outwidth, outheight, outdepth));

            EdgePadding ope = new EdgePadding(inwidth, inheight, indepth, channels, leftpad, rightpad, toppad, bottompad, frontpad, rearpad);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            ope.Execute(x_tensor, y_tensor);
            ope.Execute(x_tensor, y_tensor);
            ope.Execute(x_tensor, y_tensor);
            ope.Execute(x_tensor, y_tensor);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds / 4} msec");
        }
    }
}
