using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Connection1D {
    [TestClass]
    public class BilinearZoomTest {
        [TestMethod]
        public void ReferenceTest() {
            int scale = 2;
            int channels = 2, inwidth = 4, batch = 2;
            int outwidth = inwidth * scale;

            float[] xval = (new float[channels * inwidth * batch]).Select((_, idx) => idx * 2e-3f).ToArray();
            float[] yval = (new float[channels * outwidth * batch]).Select((_, idx) => idx * 1e-3f).ToArray();

            Tensor xtensor = new Tensor(Shape.Map1D(channels, inwidth, batch), xval);
            Tensor ytensor = new Tensor(Shape.Map1D(channels, outwidth, batch), yval);

            ParameterField x = xtensor;
            VariableField y_actual = ytensor;

            Field y_expect = BilinearZoom1D(x);
            Field err = Abs(y_expect - y_actual);

            (Flow flow, Parameters Parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradTensor.State;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            -3.333333e-04f, 6.666667e-04f, -1.000000e-03f, 0.000000e+00f, -1.000000e-03f, 8.673617e-19f, -1.666667e-03f, -6.666667e-04f,
            -3.333333e-04f, 6.666667e-04f, -1.000000e-03f, 0.000000e+00f, -1.000000e-03f, 1.734723e-18f, -1.666667e-03f, -6.666667e-04f,
        };
    }
}
