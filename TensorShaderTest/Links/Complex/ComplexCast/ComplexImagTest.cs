using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.ComplexArithmetric {
    [TestClass]
    public class ComplexImagTest {
        [TestMethod]
        public void ReferenceTest() {
            int length = 24;

            float[] xval = (new float[length]).Select((_, idx) => (float)idx * 2 - length).ToArray();
            float[] yval = (new float[length / 2]).Select((_, idx) => (float)idx / 2).ToArray();

            Tensor xtensor = new Tensor(Shape.Vector(length), xval);

            Tensor ytensor =  new Tensor(Shape.Vector(length / 2), yval);

            ParameterField x = xtensor;
            VariableField y_actual = ytensor;

            Field y_expect = ComplexImag(x);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters Parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradTensor.State;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            0.000000e+00f, -2.200000e+01f, 0.000000e+00f, -1.850000e+01f, 0.000000e+00f, -1.500000e+01f,
            0.000000e+00f, -1.150000e+01f, 0.000000e+00f, -8.000000e+00f, 0.000000e+00f, -4.500000e+00f,
            0.000000e+00f, -1.000000e+00f, 0.000000e+00f, 2.500000e+00f, 0.000000e+00f, 6.000000e+00f,
            0.000000e+00f, 9.500000e+00f, 0.000000e+00f, 1.300000e+01f, 0.000000e+00f, 1.650000e+01f,
        };
    }
}
