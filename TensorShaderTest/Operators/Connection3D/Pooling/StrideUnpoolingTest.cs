using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection3D;
using TensorShaderAvxBackend.API;

namespace TensorShaderTest.Operators.Connection3D {
    [TestClass]
    public class StrideUnpoolingTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            Random random = new Random();

            foreach (int batch in new int[] { 1, 2, 3 }) {
                foreach (int channels in new int[] { 1, 2, 3, 5 }) {
                    foreach (int stride in new int[] { 2, 3, 4 }) {
                        foreach (int indepth in new int[] { stride, 5, 7, 8, 11 }) {
                            foreach (int inheight in new int[] { stride, 5, 7, 8, 11 }) {
                                foreach (int inwidth in new int[] { stride, 5, 7, 8, 11 }) {
                                    int outwidth = inwidth / stride, outheight = inheight / stride, outdepth = indepth / stride;

                                    float[] yval = (new float[outwidth * outheight * outdepth * channels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

                                    Map3D y = new Map3D(channels, outwidth, outheight, outdepth, batch, yval);

                                    Map3D x = Reference(y, inwidth, inheight, indepth, stride);

                                    OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map3D(channels, inwidth, inheight, indepth, batch));
                                    OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map3D(channels, outwidth, outheight, outdepth, batch), yval);

                                    TensorShaderAvxBackend.Randomize.Uniform((uint)x_tensor.Length, x_tensor.Buffer, random);

                                    StrideUnpooling ope = new StrideUnpooling(inwidth, inheight, indepth, channels, stride, batch);

                                    ope.Execute(y_tensor, x_tensor);

                                    float[] x_expect = x.ToArray();
                                    float[] x_actual = x_tensor.State;

                                    CollectionAssert.AreEqual(yval, y_tensor.State);

                                    AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {channels},{stride},{inwidth},{inheight},{indepth},{batch}");

                                    Console.WriteLine($"pass: {channels},{stride},{inwidth},{inheight},{indepth},{batch}");

                                }
                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        public static Map3D Reference(Map3D y, int outw, int outh, int outd, int stride) {
            int channels = y.Channels, batch = y.Batch;
            int inw = outw / stride, inh = outh / stride, ind = outd / stride;

            Map3D x = new Map3D(channels, outw, outh, outd, batch);

            for (int th = 0; th < batch; th++) {
                for (int ix, iy, iz = 0; iz < ind; iz++) {
                    for (iy = 0; iy < inh; iy++) {
                        for (ix = 0; ix < inw; ix++) {
                            int ox = ix * stride, oy = iy * stride, oz = iz * stride;

                            for (int ch = 0; ch < channels; ch++) {
                                x[ch, ox, oy, oz, th] = y[ch, ix, iy, iz, th];
                            }
                        }
                    }
                }
            }

            return x;
        }

        [TestMethod]
        public void SpeedTest() {
            int outwidth = 64, outheight = 64, outdepth = 64, channels = 32, stride = 2, batch = 4;
            int inwidth = outwidth / stride, inheight = outheight / stride, indepth = outdepth / stride;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map3D(channels, inwidth, inheight, indepth, batch));
            OverflowCheckedTensor y_tensor = new OverflowCheckedTensor(Shape.Map3D(channels, outwidth, outheight, outdepth, batch));

            StrideUnpooling ope = new StrideUnpooling(outwidth, outheight, outdepth, channels, stride, batch);

            Cuda.Profiler.Initialize("../../../profiler.nvsetting", "../../nvprofiles/strideunpool_3d.nvvp");
            Cuda.Profiler.Start();

            ope.Execute(x_tensor, y_tensor);

            Cuda.Profiler.Stop();
        }
    }
}