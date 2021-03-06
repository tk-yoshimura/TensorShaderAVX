using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.Connection2D;

namespace TensorShaderTest.Operators.Connection2D {
    [TestClass]
    public class KernelProductTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach(int inchannels in new int[]{ 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach(int outchannels in new int[]{ 7, 13 }) {
                        foreach (int kheight in new int[] { 1, 3, 5 }) {
                            foreach (int kwidth in new int[] { 1, 3, 5 }) {
                                foreach (int stride in new int[] { 1, 2, 3 }) {
                                    foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                                        foreach (int inheight in new int[] { 8, 9, 19, 23 }) {
                                            int outwidth = (inwidth - kwidth) / stride + 1, outheight = (inheight - kheight) / stride + 1;

                                            float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                            float[] gyval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                            Map2D x = new Map2D(inchannels, inwidth, inheight, batch, xval);
                                            Map2D gy = new Map2D(outchannels, outwidth, outheight, batch, gyval);

                                            Filter2D gw = Reference(x, gy, kwidth, kheight, stride);

                                            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(inchannels, inwidth, inheight, batch), xval);
                                            OverflowCheckedTensor gy_tensor = new OverflowCheckedTensor(Shape.Map2D(outchannels, outwidth, outheight, batch), gyval);

                                            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel2D(inchannels, outchannels, kwidth, kheight));

                                            KernelProduct ope = new KernelProduct(inwidth, inheight, inchannels, outchannels, kwidth, kheight, stride, batch);

                                            ope.Execute(x_tensor, gy_tensor, gw_tensor);

                                            float[] gw_expect = gw.ToArray();
                                            float[] gw_actual = gw_tensor.State;

                                            CollectionAssert.AreEqual(xval, x_tensor.State);
                                            CollectionAssert.AreEqual(gyval, gy_tensor.State);

                                            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{stride},{inwidth},{inheight},{batch}");

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
        public void SpeedTest() {
            int inwidth = 512, inheight = 512, inchannels = 32, outchannels = 32, ksize = 3, stride = 2;
            int outwidth = (inwidth - ksize) / stride + 1, outheight = (inheight - ksize) / stride + 1;

            OverflowCheckedTensor x_tensor = new OverflowCheckedTensor(Shape.Map2D(inchannels, inwidth, inheight));
            OverflowCheckedTensor gy_tensor = new OverflowCheckedTensor(Shape.Map2D(outchannels, outwidth, outheight));

            OverflowCheckedTensor gw_tensor = new OverflowCheckedTensor(Shape.Kernel2D(inchannels, outchannels, ksize, ksize));

            KernelProduct ope = new KernelProduct(inwidth, inheight, inchannels, outchannels, ksize, ksize, stride);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            ope.Execute(x_tensor, gy_tensor, gw_tensor);
            ope.Execute(x_tensor, gy_tensor, gw_tensor);
            ope.Execute(x_tensor, gy_tensor, gw_tensor);
            ope.Execute(x_tensor, gy_tensor, gw_tensor);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds / 4} msec");
        }

        public static Filter2D Reference(Map2D x, Map2D gy, int kwidth, int kheight, int stride) {
            int inchannels = x.Channels, outchannels = gy.Channels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, outw = gy.Width, outh = gy.Height;

            if (outw != (inw - kwidth) / stride + 1 || outh != (inh - kheight) / stride + 1) {
                throw new ArgumentException("mismatch shape");
            }

            Filter2D w = new Filter2D(inchannels, outchannels, kwidth, kheight);

            for(int kx, ky = 0; ky < kheight; ky++) {
                for(kx = 0; kx < kwidth; kx++) {
                    for(int th = 0; th < batch; th++) {
                        for(int inch, outch = 0; outch < outchannels; outch++) {
                            for(inch = 0; inch < inchannels; inch++) {
                                double sum = 0;

                                for(int ix, iy = ky, ox, oy = 0; oy < outh; iy += stride, oy++) {
                                    for(ix = kx, ox = 0; ox < outw; ix += stride, ox++) {
                                        sum += x[inch, ix, iy, th] * gy[outch, ox, oy, th];
                                    }
                                }

                                w[inch, outch, kx, ky] += sum;
                            }
                        }

                    }
                }
            }

            return w;
        }

        public static Filter2D OptimizedReference(Map2D x, Map2D gy, int kwidth, int kheight, int stride) {
            int inchannels = x.Channels, outchannels = gy.Channels, batch = x.Batch;
            int inw = x.Width, inh = x.Height, outw = gy.Width, outh = gy.Height;

            if (outw < (inw - kwidth) / stride + 1 || outh < (inh - kheight) / stride + 1) {
                throw new ArgumentException("mismatch shape");
            }

            Filter2D w = new Filter2D(outchannels, inchannels, kwidth, kheight);

            for(int kx, ky = 0; ky < kheight; ky++) {
                for(kx = 0; kx < kwidth; kx++) {
                    for(int th = 0; th < batch; th++) {
                        for(int outch, inch = 0; inch < inchannels; inch++) {
                            for(outch = 0; outch < outchannels; outch++) {
                                int filter_idx = inch + inchannels * outch + (kx + ky * kwidth) * inchannels * outchannels;
                                int inmap_org = inch + (kx + ky * inw) * inchannels + th * inw * inh * inchannels;
                                int outmap_idx = outch + th * outw * outh * outchannels;

                                double sum = 0;

                                for(int ox, oy = 0; oy < outh; oy++) {
                                    int inmap_idx = inmap_org;

                                    for(ox = 0; ox < outw; ox++) {
                                        sum += x[inmap_idx] * gy[outmap_idx];

                                        inmap_idx += inchannels * stride;
                                        outmap_idx += outchannels;
                                    }

                                    inmap_org += inchannels * inw * stride;
                                }

                                w[filter_idx] += sum;
                            }
                        }
                    }
                }
            }

            return w;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 7, outchannels = 11, kwidth = 3, kheight = 5, stride = 2, inwidth = 13, inheight = 17;
            int outwidth = (inwidth - kwidth) / stride + 1, outheight = (inheight - kheight) / stride + 1;

            float[] xval = (new float[inwidth * inheight * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] gyval = (new float[outwidth * outheight * outchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map2D x = new Map2D(inchannels, inwidth, inheight, 1, xval);
            Map2D gy = new Map2D(outchannels, outwidth, outheight, 1, gyval);

            Filter2D gw = Reference(x, gy, kwidth, kheight, stride);

            float[] gw_expect = {
                3.709790230e+00f, 3.719681501e+00f, 3.729572058e+00f, 3.739463568e+00f, 3.749354362e+00f,
                3.759245396e+00f, 3.769136190e+00f, 3.685388803e+00f, 3.695237398e+00f, 3.705086470e+00f,
                3.714935303e+00f, 3.724784374e+00f, 3.734634161e+00f, 3.744482517e+00f, 3.660985947e+00f,
                3.670793295e+00f, 3.680600166e+00f, 3.690407515e+00f, 3.700213671e+00f, 3.710021734e+00f,
                3.719828367e+00f, 3.636584997e+00f, 3.646349430e+00f, 3.656113863e+00f, 3.665879965e+00f,
                3.675644636e+00f, 3.685408831e+00f, 3.695174456e+00f, 3.612182856e+00f, 3.621904850e+00f,
                3.631627798e+00f, 3.641351700e+00f, 3.651074171e+00f, 3.660797834e+00f, 3.670520782e+00f,
                3.587780714e+00f, 3.597460747e+00f, 3.607141972e+00f, 3.616822958e+00f, 3.626504183e+00f,
                3.636185408e+00f, 3.645866394e+00f, 3.563378334e+00f, 3.573017597e+00f, 3.582656622e+00f,
                3.592295647e+00f, 3.601934433e+00f, 3.611573458e+00f, 3.621212006e+00f, 3.538975716e+00f,
                3.548573256e+00f, 3.558171034e+00f, 3.567767143e+00f, 3.577364445e+00f, 3.586961269e+00f,
                3.596558094e+00f, 3.514574051e+00f, 3.524129391e+00f, 3.533683777e+00f, 3.543239117e+00f,
                3.552793980e+00f, 3.562349558e+00f, 3.571904898e+00f, 3.490172386e+00f, 3.499685049e+00f,
                3.509198427e+00f, 3.518711567e+00f, 3.528224468e+00f, 3.537736654e+00f, 3.547250271e+00f,
                3.465770721e+00f, 3.475241423e+00f, 3.484711885e+00f, 3.494182825e+00f, 3.503654242e+00f,
                3.513125181e+00f, 3.522597075e+00f, 3.779026747e+00f, 3.788918972e+00f, 3.798809767e+00f,
                3.808700562e+00f, 3.818591118e+00f, 3.828482866e+00f, 3.838372946e+00f, 3.754331827e+00f,
                3.764180660e+00f, 3.774029493e+00f, 3.783878803e+00f, 3.793727636e+00f, 3.803576231e+00f,
                3.813425779e+00f, 3.729635477e+00f, 3.739442348e+00f, 3.749249697e+00f, 3.759056568e+00f,
                3.768862963e+00f, 3.778670311e+00f, 3.788476944e+00f, 3.704939604e+00f, 3.714704514e+00f,
                3.724469662e+00f, 3.734234095e+00f, 3.743999481e+00f, 3.753764629e+00f, 3.763529539e+00f,
                3.680243254e+00f, 3.689966440e+00f, 3.699688911e+00f, 3.709412575e+00f, 3.719135046e+00f,
                3.728858471e+00f, 3.738580942e+00f, 3.655547142e+00f, 3.665228367e+00f, 3.674909592e+00f,
                3.684590340e+00f, 3.694270849e+00f, 3.703952074e+00f, 3.713632822e+00f, 3.630851030e+00f,
                3.640490055e+00f, 3.650129318e+00f, 3.659768581e+00f, 3.669407368e+00f, 3.679045916e+00f,
                3.688685179e+00f, 3.606155634e+00f, 3.615751982e+00f, 3.625349045e+00f, 3.634946346e+00f,
                3.644543171e+00f, 3.654140234e+00f, 3.663737535e+00f, 3.581459284e+00f, 3.591014624e+00f,
                3.600569487e+00f, 3.610124588e+00f, 3.619679451e+00f, 3.629234314e+00f, 3.638789177e+00f,
                3.556763172e+00f, 3.566275835e+00f, 3.575789928e+00f, 3.585302591e+00f, 3.594815016e+00f,
                3.604327917e+00f, 3.613841057e+00f, 3.532066822e+00f, 3.541538000e+00f, 3.551009178e+00f,
                3.560480595e+00f, 3.569951057e+00f, 3.579421997e+00f, 3.588893175e+00f, 3.848264456e+00f,
                3.858155489e+00f, 3.868046761e+00f, 3.877937794e+00f, 3.887828112e+00f, 3.897719622e+00f,
                3.907610655e+00f, 3.823274612e+00f, 3.833123684e+00f, 3.842972517e+00f, 3.852821350e+00f,
                3.862670183e+00f, 3.872519016e+00f, 3.882368803e+00f, 3.798284531e+00f, 3.808091879e+00f,
                3.817898273e+00f, 3.827705622e+00f, 3.837512970e+00f, 3.847319603e+00f, 3.857126474e+00f,
                3.773294687e+00f, 3.783059120e+00f, 3.792824268e+00f, 3.802589893e+00f, 3.812354326e+00f,
                3.822119713e+00f, 3.831884384e+00f, 3.748304605e+00f, 3.758027554e+00f, 3.767750263e+00f,
                3.777473211e+00f, 3.787196159e+00f, 3.796919584e+00f, 3.806642294e+00f, 3.723314524e+00f,
                3.732995510e+00f, 3.742675781e+00f, 3.752357006e+00f, 3.762038946e+00f, 3.771719217e+00f,
                3.781400681e+00f, 3.698324442e+00f, 3.707963943e+00f, 3.717602491e+00f, 3.727241516e+00f,
                3.736880064e+00f, 3.746519566e+00f, 3.756158590e+00f, 3.673334599e+00f, 3.682931185e+00f,
                3.692528248e+00f, 3.702125311e+00f, 3.711722851e+00f, 3.721319675e+00f, 3.730916739e+00f,
                3.648344278e+00f, 3.657899380e+00f, 3.667454004e+00f, 3.677009344e+00f, 3.686563969e+00f,
                3.696119785e+00f, 3.705674410e+00f, 3.623354435e+00f, 3.632867336e+00f, 3.642380238e+00f,
                3.651893139e+00f, 3.661406040e+00f, 3.670918703e+00f, 3.680432796e+00f, 3.598364115e+00f,
                3.607836008e+00f, 3.617305994e+00f, 3.626777649e+00f, 3.636248112e+00f, 3.645719290e+00f,
                3.655190945e+00f, 4.609871864e+00f, 4.619761944e+00f, 4.629652977e+00f, 4.639544487e+00f,
                4.649435520e+00f, 4.659327030e+00f, 4.669216633e+00f, 4.581647396e+00f, 4.591495991e+00f,
                4.601345539e+00f, 4.611194134e+00f, 4.621043682e+00f, 4.630891800e+00f, 4.640741348e+00f,
                4.553423882e+00f, 4.563230515e+00f, 4.573037148e+00f, 4.582844257e+00f, 4.592651367e+00f,
                4.602458477e+00f, 4.612265587e+00f, 4.525199890e+00f, 4.534965038e+00f, 4.544729233e+00f,
                4.554494381e+00f, 4.564260483e+00f, 4.574023724e+00f, 4.583789349e+00f, 4.496974468e+00f,
                4.506698132e+00f, 4.516421318e+00f, 4.526144505e+00f, 4.535868645e+00f, 4.545591354e+00f,
                4.555314064e+00f, 4.468752384e+00f, 4.478432178e+00f, 4.488113880e+00f, 4.497794628e+00f,
                4.507475853e+00f, 4.517156124e+00f, 4.526837349e+00f, 4.440527439e+00f, 4.450166702e+00f,
                4.459804535e+00f, 4.469443798e+00f, 4.479083061e+00f, 4.488722801e+00f, 4.498360634e+00f,
                4.412302971e+00f, 4.421900749e+00f, 4.431498051e+00f, 4.441094398e+00f, 4.450691223e+00f,
                4.460288525e+00f, 4.469885826e+00f, 4.384078503e+00f, 4.393634319e+00f, 4.403189659e+00f,
                4.412744999e+00f, 4.422299862e+00f, 4.431854248e+00f, 4.441409588e+00f, 4.355854511e+00f,
                4.365368366e+00f, 4.374881744e+00f, 4.384394169e+00f, 4.393907070e+00f, 4.403419971e+00f,
                4.412933350e+00f, 4.327630520e+00f, 4.337102890e+00f, 4.346573353e+00f, 4.356044769e+00f,
                4.365515232e+00f, 4.374985695e+00f, 4.384456635e+00f, 4.679108620e+00f, 4.688998699e+00f,
                4.698890209e+00f, 4.708782196e+00f, 4.718672276e+00f, 4.728563786e+00f, 4.738454342e+00f,
                4.650590420e+00f, 4.660439491e+00f, 4.670288086e+00f, 4.680137157e+00f, 4.689986229e+00f,
                4.699835777e+00f, 4.709683895e+00f, 4.622072697e+00f, 4.631880283e+00f, 4.641687393e+00f,
                4.651494503e+00f, 4.661299706e+00f, 4.671107292e+00f, 4.680914402e+00f, 4.593554020e+00f,
                4.603319168e+00f, 4.613083839e+00f, 4.622850418e+00f, 4.632615089e+00f, 4.642379284e+00f,
                4.652144432e+00f, 4.565036774e+00f, 4.574759960e+00f, 4.584482670e+00f, 4.594205379e+00f,
                4.603928566e+00f, 4.613651276e+00f, 4.623374939e+00f, 4.536519527e+00f, 4.546199322e+00f,
                4.555880070e+00f, 4.565561771e+00f, 4.575242519e+00f, 4.584922791e+00f, 4.594604015e+00f,
                4.508000851e+00f, 4.517639637e+00f, 4.527278900e+00f, 4.536918640e+00f, 4.546556473e+00f,
                4.556195736e+00f, 4.565834999e+00f, 4.479482174e+00f, 4.489078999e+00f, 4.498676777e+00f,
                4.508272648e+00f, 4.517870426e+00f, 4.527467251e+00f, 4.537064552e+00f, 4.450965405e+00f,
                4.460519791e+00f, 4.470074177e+00f, 4.479628563e+00f, 4.489185333e+00f, 4.498740196e+00f,
                4.508293629e+00f, 4.422446728e+00f, 4.431960106e+00f, 4.441472530e+00f, 4.450985432e+00f,
                4.460497856e+00f, 4.470011234e+00f, 4.479524136e+00f, 4.393928528e+00f, 4.403399467e+00f,
                4.412868977e+00f, 4.422342777e+00f, 4.431812763e+00f, 4.441283703e+00f, 4.450754642e+00f,
                4.748344898e+00f, 4.758236885e+00f, 4.768127441e+00f, 4.778017998e+00f, 4.787909508e+00f,
                4.797800541e+00f, 4.807691574e+00f, 4.719534397e+00f, 4.729382992e+00f, 4.739232063e+00f,
                4.749080658e+00f, 4.758930206e+00f, 4.768778324e+00f, 4.778626919e+00f, 4.690720558e+00f,
                4.700528622e+00f, 4.710335255e+00f, 4.720141411e+00f, 4.729949951e+00f, 4.739756584e+00f,
                4.749562740e+00f, 4.661909580e+00f, 4.671674252e+00f, 4.681439877e+00f, 4.691204548e+00f,
                4.700969219e+00f, 4.710733891e+00f, 4.720499992e+00f, 4.633098125e+00f, 4.642820358e+00f,
                4.652544022e+00f, 4.662266254e+00f, 4.671988964e+00f, 4.681712627e+00f, 4.691435814e+00f,
                4.604284763e+00f, 4.613965988e+00f, 4.623647213e+00f, 4.633328438e+00f, 4.643009663e+00f,
                4.652690411e+00f, 4.662371159e+00f, 4.575472355e+00f, 4.585112572e+00f, 4.594751835e+00f,
                4.604390621e+00f, 4.614029884e+00f, 4.623667717e+00f, 4.633307934e+00f, 4.546661377e+00f,
                4.556259632e+00f, 4.565855503e+00f, 4.575451851e+00f, 4.585049152e+00f, 4.594646454e+00f,
                4.604243755e+00f, 4.517849445e+00f, 4.527404785e+00f, 4.536960125e+00f, 4.546514034e+00f,
                4.556069851e+00f, 4.565624237e+00f, 4.575180054e+00f, 4.489036560e+00f, 4.498550415e+00f,
                4.508062840e+00f, 4.517576218e+00f, 4.527089119e+00f, 4.536603451e+00f, 4.546115875e+00f,
                4.460225582e+00f, 4.469695568e+00f, 4.479167461e+00f, 4.488637447e+00f, 4.498109341e+00f,
                4.507579803e+00f, 4.517051697e+00f, 5.509952068e+00f, 5.519843102e+00f, 5.529735088e+00f,
                5.539626598e+00f, 5.549517155e+00f, 5.559407711e+00f, 5.569299221e+00f, 5.477907181e+00f,
                5.487755299e+00f, 5.497604370e+00f, 5.507453918e+00f, 5.517302036e+00f, 5.527151108e+00f,
                5.537001133e+00f, 5.445860386e+00f, 5.455667496e+00f, 5.465475082e+00f, 5.475280762e+00f,
                5.485087872e+00f, 5.494894505e+00f, 5.504702568e+00f, 5.413815022e+00f, 5.423580170e+00f,
                5.433343410e+00f, 5.443109989e+00f, 5.452874184e+00f, 5.462639809e+00f, 5.472405434e+00f,
                5.381768227e+00f, 5.391490936e+00f, 5.401214123e+00f, 5.410937309e+00f, 5.420660019e+00f,
                5.430383205e+00f, 5.440106392e+00f, 5.349722385e+00f, 5.359402657e+00f, 5.369084358e+00f,
                5.378765583e+00f, 5.388447285e+00f, 5.398127079e+00f, 5.407809258e+00f, 5.317675591e+00f,
                5.327315807e+00f, 5.336954594e+00f, 5.346594334e+00f, 5.356233120e+00f, 5.365871429e+00f,
                5.375510693e+00f, 5.285630226e+00f, 5.295227051e+00f, 5.304823875e+00f, 5.314422607e+00f,
                5.324018478e+00f, 5.333615303e+00f, 5.343212605e+00f, 5.253583908e+00f, 5.263140202e+00f,
                5.272694588e+00f, 5.282248974e+00f, 5.291804314e+00f, 5.301359653e+00f, 5.310914516e+00f,
                5.221538544e+00f, 5.231052399e+00f, 5.240565300e+00f, 5.250077724e+00f, 5.259590626e+00f,
                5.269102097e+00f, 5.278616428e+00f, 5.189492702e+00f, 5.198963165e+00f, 5.208433628e+00f,
                5.217905045e+00f, 5.227376461e+00f, 5.236847401e+00f, 5.246319294e+00f, 5.579188824e+00f,
                5.589081287e+00f, 5.598970890e+00f, 5.608863354e+00f, 5.618752480e+00f, 5.628643513e+00f,
                5.638535023e+00f, 5.546849728e+00f, 5.556697845e+00f, 5.566546917e+00f, 5.576396942e+00f,
                5.586246967e+00f, 5.596094608e+00f, 5.605944633e+00f, 5.514508724e+00f, 5.524317265e+00f,
                5.534123898e+00f, 5.543931007e+00f, 5.553736687e+00f, 5.563543797e+00f, 5.573351860e+00f,
                5.482169628e+00f, 5.491934299e+00f, 5.501699448e+00f, 5.511464596e+00f, 5.521229267e+00f,
                5.530994415e+00f, 5.540759563e+00f, 5.449830532e+00f, 5.459553242e+00f, 5.469275475e+00f,
                5.478998184e+00f, 5.488720894e+00f, 5.498444557e+00f, 5.508168221e+00f, 5.417489052e+00f,
                5.427170277e+00f, 5.436851025e+00f, 5.446532726e+00f, 5.456212521e+00f, 5.465894222e+00f,
                5.475575924e+00f, 5.385149479e+00f, 5.394788265e+00f, 5.404427528e+00f, 5.414066792e+00f,
                5.423706532e+00f, 5.433343410e+00f, 5.442982674e+00f, 5.352808952e+00f, 5.362407207e+00f,
                5.372003078e+00f, 5.381601810e+00f, 5.391198158e+00f, 5.400794983e+00f, 5.410391331e+00f,
                5.320471287e+00f, 5.330024242e+00f, 5.339579582e+00f, 5.349134922e+00f, 5.358689308e+00f,
                5.368243694e+00f, 5.377799034e+00f, 5.288128853e+00f, 5.297641754e+00f, 5.307154655e+00f,
                5.316668987e+00f, 5.326180935e+00f, 5.335694313e+00f, 5.345208168e+00f, 5.255790234e+00f,
                5.265260220e+00f, 5.274731159e+00f, 5.284202576e+00f, 5.293673515e+00f, 5.303144932e+00f,
                5.312614918e+00f, 5.648427010e+00f, 5.658318520e+00f, 5.668208122e+00f, 5.678099155e+00f,
                5.687990189e+00f, 5.697882175e+00f, 5.707771301e+00f, 5.615791321e+00f, 5.625641346e+00f,
                5.635489941e+00f, 5.645339966e+00f, 5.655189037e+00f, 5.665038586e+00f, 5.674886703e+00f,
                5.583159447e+00f, 5.592965603e+00f, 5.602773666e+00f, 5.612579346e+00f, 5.622386932e+00f,
                5.632194042e+00f, 5.642000198e+00f, 5.550524235e+00f, 5.560289383e+00f, 5.570055962e+00f,
                5.579820156e+00f, 5.589582920e+00f, 5.599350929e+00f, 5.609115124e+00f, 5.517891407e+00f,
                5.527613163e+00f, 5.537335873e+00f, 5.547059536e+00f, 5.556782722e+00f, 5.566506863e+00f,
                5.576228619e+00f, 5.485256672e+00f, 5.494936943e+00f, 5.504619598e+00f, 5.514298916e+00f,
                5.523979187e+00f, 5.533662319e+00f, 5.543342590e+00f, 5.452623844e+00f, 5.462261677e+00f,
                5.471901417e+00f, 5.481540680e+00f, 5.491179466e+00f, 5.500817776e+00f, 5.510456085e+00f,
                5.419989109e+00f, 5.429585934e+00f, 5.439182758e+00f, 5.448780060e+00f, 5.458374500e+00f,
                5.467972279e+00f, 5.477571964e+00f, 5.387354374e+00f, 5.396909237e+00f, 5.406465530e+00f,
                5.416019440e+00f, 5.425573826e+00f, 5.435129642e+00f, 5.444684029e+00f, 5.354720592e+00f,
                5.364233971e+00f, 5.373746872e+00f, 5.383259296e+00f, 5.392772675e+00f, 5.402285099e+00f,
                5.411798477e+00f, 5.322086334e+00f, 5.331557751e+00f, 5.341029167e+00f, 5.350500107e+00f,
                5.359971046e+00f, 5.369441032e+00f, 5.378911972e+00f, 6.410033703e+00f, 6.419924736e+00f,
                6.429814816e+00f, 6.439707279e+00f, 6.449597836e+00f, 6.459489822e+00f, 6.469379902e+00f,
                6.374166012e+00f, 6.384015560e+00f, 6.393863201e+00f, 6.403712273e+00f, 6.413561821e+00f,
                6.423410416e+00f, 6.433259964e+00f, 6.338298321e+00f, 6.348105431e+00f, 6.357911587e+00f,
                6.367720127e+00f, 6.377524853e+00f, 6.387333870e+00f, 6.397139072e+00f, 6.302429199e+00f,
                6.312193871e+00f, 6.321960449e+00f, 6.331724644e+00f, 6.341490746e+00f, 6.351253510e+00f,
                6.361018658e+00f, 6.266561508e+00f, 6.276285172e+00f, 6.286008835e+00f, 6.295730591e+00f,
                6.305454731e+00f, 6.315176487e+00f, 6.324900150e+00f, 6.230693340e+00f, 6.240374088e+00f,
                6.250054836e+00f, 6.259737015e+00f, 6.269416809e+00f, 6.279097557e+00f, 6.288779259e+00f,
                6.194825172e+00f, 6.204463482e+00f, 6.214104652e+00f, 6.223742008e+00f, 6.233382225e+00f,
                6.243020535e+00f, 6.252659321e+00f, 6.158957958e+00f, 6.168553829e+00f, 6.178151608e+00f,
                6.187748909e+00f, 6.197345257e+00f, 6.206943989e+00f, 6.216538906e+00f, 6.123091221e+00f,
                6.132644653e+00f, 6.142199993e+00f, 6.151753902e+00f, 6.161309242e+00f, 6.170865059e+00f,
                6.180418968e+00f, 6.087221622e+00f, 6.096735001e+00f, 6.106246948e+00f, 6.115760326e+00f,
                6.125273705e+00f, 6.134786606e+00f, 6.144298553e+00f, 6.051354408e+00f, 6.060824394e+00f,
                6.070294857e+00f, 6.079767704e+00f, 6.089238644e+00f, 6.098708630e+00f, 6.108179569e+00f,
                6.479270935e+00f, 6.489161015e+00f, 6.499052048e+00f, 6.508944035e+00f, 6.518834114e+00f,
                6.528725624e+00f, 6.538617134e+00f, 6.443109512e+00f, 6.452958584e+00f, 6.462806702e+00f,
                6.472654819e+00f, 6.482504368e+00f, 6.492352962e+00f, 6.502202988e+00f, 6.406945705e+00f,
                6.416754246e+00f, 6.426560879e+00f, 6.436367512e+00f, 6.446175098e+00f, 6.455981255e+00f,
                6.465788364e+00f, 6.370785236e+00f, 6.380548954e+00f, 6.390315056e+00f, 6.400079250e+00f,
                6.409844398e+00f, 6.419609547e+00f, 6.429375172e+00f, 6.334623337e+00f, 6.344345570e+00f,
                6.354068756e+00f, 6.363791466e+00f, 6.373515606e+00f, 6.383237362e+00f, 6.392961502e+00f,
                6.298460484e+00f, 6.308141708e+00f, 6.317821980e+00f, 6.327503204e+00f, 6.337183952e+00f,
                6.346866131e+00f, 6.356547356e+00f, 6.262298584e+00f, 6.271937847e+00f, 6.281577587e+00f,
                6.291215420e+00f, 6.300853729e+00f, 6.310493469e+00f, 6.320133209e+00f, 6.226137161e+00f,
                6.235734463e+00f, 6.245331287e+00f, 6.254927635e+00f, 6.264523983e+00f, 6.274121761e+00f,
                6.283718586e+00f, 6.189975262e+00f, 6.199530125e+00f, 6.209084988e+00f, 6.218638897e+00f,
                6.228194237e+00f, 6.237749100e+00f, 6.247304440e+00f, 6.153812885e+00f, 6.163325787e+00f,
                6.172838211e+00f, 6.182350636e+00f, 6.191864491e+00f, 6.201378822e+00f, 6.210891247e+00f,
                6.117650986e+00f, 6.127120972e+00f, 6.136591434e+00f, 6.146063328e+00f, 6.155533791e+00f,
                6.165004730e+00f, 6.174475670e+00f, 6.548507690e+00f, 6.558399200e+00f, 6.568289280e+00f,
                6.578180313e+00f, 6.588072300e+00f, 6.597963810e+00f, 6.607853889e+00f, 6.512051105e+00f,
                6.521899223e+00f, 6.531749725e+00f, 6.541599274e+00f, 6.551447868e+00f, 6.561296463e+00f,
                6.571146488e+00f, 6.475595474e+00f, 6.485401630e+00f, 6.495210171e+00f, 6.505015850e+00f,
                6.514823914e+00f, 6.524630070e+00f, 6.534438610e+00f, 6.439138889e+00f, 6.448905468e+00f,
                6.458669662e+00f, 6.468434811e+00f, 6.478199482e+00f, 6.487963676e+00f, 6.497728348e+00f,
                6.402684212e+00f, 6.412406445e+00f, 6.422129154e+00f, 6.431852341e+00f, 6.441576004e+00f,
                6.451299667e+00f, 6.461022377e+00f, 6.366227150e+00f, 6.375908852e+00f, 6.385589600e+00f,
                6.395269871e+00f, 6.404953003e+00f, 6.414632320e+00f, 6.424313545e+00f, 6.329770565e+00f,
                6.339409828e+00f, 6.349049568e+00f, 6.358688354e+00f, 6.368328094e+00f, 6.377966404e+00f,
                6.387604713e+00f, 6.293315411e+00f, 6.302912712e+00f, 6.312510014e+00f, 6.322106361e+00f,
                6.331703663e+00f, 6.341300011e+00f, 6.350897789e+00f, 6.256859303e+00f, 6.266415596e+00f,
                6.275969982e+00f, 6.285525322e+00f, 6.295079708e+00f, 6.304634094e+00f, 6.314189911e+00f,
                6.220404148e+00f, 6.229917049e+00f, 6.239429474e+00f, 6.248941898e+00f, 6.258454800e+00f,
                6.267968178e+00f, 6.277480602e+00f, 6.183948040e+00f, 6.193417549e+00f, 6.202888489e+00f,
                6.212359905e+00f, 6.221832275e+00f, 6.231302261e+00f, 6.240773678e+00f, 7.310114384e+00f,
                7.320005894e+00f, 7.329896450e+00f, 7.339787006e+00f, 7.349678993e+00f, 7.359569073e+00f,
                7.369462013e+00f, 7.270424843e+00f, 7.280272961e+00f, 7.290122509e+00f, 7.299973011e+00f,
                7.309821606e+00f, 7.319670677e+00f, 7.329518795e+00f, 7.230735302e+00f, 7.240541935e+00f,
                7.250349045e+00f, 7.260155678e+00f, 7.269962311e+00f, 7.279768944e+00f, 7.289576530e+00f,
                7.191043854e+00f, 7.200809002e+00f, 7.210575104e+00f, 7.220339298e+00f, 7.230104923e+00f,
                7.239868641e+00f, 7.249633789e+00f, 7.151355743e+00f, 7.161078453e+00f, 7.170800686e+00f,
                7.180522919e+00f, 7.190245628e+00f, 7.199969769e+00f, 7.209693432e+00f, 7.111664295e+00f,
                7.121345043e+00f, 7.131026268e+00f, 7.140707016e+00f, 7.150388241e+00f, 7.160068989e+00f,
                7.169750214e+00f, 7.071973324e+00f, 7.081613541e+00f, 7.091252804e+00f, 7.100891590e+00f,
                7.110531807e+00f, 7.120168209e+00f, 7.129807949e+00f, 7.032284737e+00f, 7.041882992e+00f,
                7.051477909e+00f, 7.061076641e+00f, 7.070672512e+00f, 7.080269337e+00f, 7.089868069e+00f,
                6.992595673e+00f, 7.002149105e+00f, 7.011704445e+00f, 7.021259308e+00f, 7.030814648e+00f,
                7.040369034e+00f, 7.049924850e+00f, 6.952904224e+00f, 6.962417126e+00f, 6.971930981e+00f,
                6.981443882e+00f, 6.990956306e+00f, 7.000469208e+00f, 7.009983540e+00f, 6.913215637e+00f,
                6.922685146e+00f, 6.932156563e+00f, 6.941628456e+00f, 6.951097965e+00f, 6.960570812e+00f,
                6.970040798e+00f, 7.379352093e+00f, 7.389242649e+00f, 7.399133205e+00f, 7.409023762e+00f,
                7.418915272e+00f, 7.428806782e+00f, 7.438697815e+00f, 7.339367867e+00f, 7.349215508e+00f,
                7.359065056e+00f, 7.368914127e+00f, 7.378763676e+00f, 7.388613701e+00f, 7.398462772e+00f,
                7.299383163e+00f, 7.309191704e+00f, 7.318997860e+00f, 7.328804970e+00f, 7.338611603e+00f,
                7.348418236e+00f, 7.358225822e+00f, 7.259397984e+00f, 7.269164085e+00f, 7.278929234e+00f,
                7.288694382e+00f, 7.298459053e+00f, 7.308225155e+00f, 7.317989349e+00f, 7.219415665e+00f,
                7.229137897e+00f, 7.238860607e+00f, 7.248584747e+00f, 7.258306980e+00f, 7.268030643e+00f,
                7.277752876e+00f, 7.179431438e+00f, 7.189113617e+00f, 7.198794365e+00f, 7.208476067e+00f,
                7.218154430e+00f, 7.227836132e+00f, 7.237517834e+00f, 7.139448643e+00f, 7.149086475e+00f,
                7.158726215e+00f, 7.168365002e+00f, 7.178003311e+00f, 7.187644005e+00f, 7.197280884e+00f,
                7.099462986e+00f, 7.109060287e+00f, 7.118657589e+00f, 7.128254414e+00f, 7.137851238e+00f,
                7.147448063e+00f, 7.157047272e+00f, 7.059479713e+00f, 7.069035053e+00f, 7.078590393e+00f,
                7.088144302e+00f, 7.097700119e+00f, 7.107254982e+00f, 7.116809368e+00f, 7.019495010e+00f,
                7.029008389e+00f, 7.038522243e+00f, 7.048034191e+00f, 7.057547569e+00f, 7.067060471e+00f,
                7.076572895e+00f, 6.979512215e+00f, 6.988981724e+00f, 6.998453617e+00f, 7.007925034e+00f,
                7.017397404e+00f, 7.026866436e+00f, 7.036336422e+00f, 7.448588371e+00f, 7.458479404e+00f,
                7.468370914e+00f, 7.478261471e+00f, 7.488152504e+00f, 7.498044014e+00f, 7.507934570e+00f,
                7.408310413e+00f, 7.418160439e+00f, 7.428009033e+00f, 7.437857151e+00f, 7.447706699e+00f,
                7.457555771e+00f, 7.467406273e+00f, 7.368032932e+00f, 7.377840042e+00f, 7.387647152e+00f,
                7.397453785e+00f, 7.407261372e+00f, 7.417068005e+00f, 7.426874161e+00f, 7.327754021e+00f,
                7.337518692e+00f, 7.347285748e+00f, 7.357049465e+00f, 7.366815567e+00f, 7.376579762e+00f,
                7.386345387e+00f, 7.287477016e+00f, 7.297200680e+00f, 7.306922913e+00f, 7.316645622e+00f,
                7.326368332e+00f, 7.336091995e+00f, 7.345815182e+00f, 7.247199059e+00f, 7.256881237e+00f,
                7.266560078e+00f, 7.276241779e+00f, 7.285922527e+00f, 7.295604706e+00f, 7.305284977e+00f,
                7.206920624e+00f, 7.216559410e+00f, 7.226199150e+00f, 7.235837460e+00f, 7.245475292e+00f,
                7.255115032e+00f, 7.264755249e+00f, 7.166642189e+00f, 7.176240444e+00f, 7.185835361e+00f,
                7.195434093e+00f, 7.205030918e+00f, 7.214627266e+00f, 7.224224567e+00f, 7.126363277e+00f,
                7.135920048e+00f, 7.145475388e+00f, 7.155028820e+00f, 7.164584637e+00f, 7.174140453e+00f,
                7.183694839e+00f, 7.086086273e+00f, 7.095599174e+00f, 7.105113029e+00f, 7.114624977e+00f,
                7.124139309e+00f, 7.133652687e+00f, 7.143164635e+00f, 7.045809746e+00f, 7.055279732e+00f,
                7.064750671e+00f, 7.074221134e+00f, 7.083692551e+00f, 7.093163967e+00f, 7.102634907e+00f,
            };

            float[] gw_actual = gw.ToArray();

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{stride},{inwidth},{inheight}");
        }

        [TestMethod]
        public void OptimizeTest() {
            float max_err = 0;

            foreach (int batch in new int[] { 1, 2 }) {
                foreach(int inchannels in new int[]{ 1, 2, 3, 4, 5, 10, 15, 20 }) {
                    foreach(int outchannels in new int[]{ 7, 13 }) {
                        foreach (int kheight in new int[] { 1, 3, 5 }) {
                            foreach (int kwidth in new int[] { 1, 3, 5 }) {
                                foreach (int stride in new int[] { 1, 2, 3 }) {
                                    foreach (int inwidth in new int[] { 8, 9, 13, 17 }) {
                                        foreach (int inheight in new int[] { 8, 9, 19, 23 }) {
                                            int outwidth = (inwidth - kwidth) / stride + 1, outheight = (inheight - kheight) / stride + 1;

                                            float[] xval = (new float[inwidth * inheight * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
                                            float[] gyval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                            Map2D x = new Map2D(inchannels, inwidth, inheight, batch, xval);
                                            Map2D gy = new Map2D(outchannels, outwidth, outheight, batch, gyval);

                                            Filter2D gw = Reference(x, gy, kwidth, kheight, stride);
                                            Filter2D gw_optimized = OptimizedReference(x, gy, kwidth, kheight, stride);

                                            float[] gw_expect = gw.ToArray();
                                            float[] gw_actual = gw_optimized.ToArray();

                                            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, ref max_err, $"mismatch value {inchannels},{outchannels},{kwidth},{kheight},{stride},{inwidth},{inheight},{batch}");

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
    }
}
