using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Layers;

namespace TensorShaderTest.Layers {
    [TestClass]
    public class ComplexDeconvolution3DTest {
        [TestMethod]
        public void ExecuteTest() {
            int inchannels = 4, outchannels = 6, inwidth = 13, inheight = 17, indepth = 19, kwidth = 3, kheight = 5, kdepth = 7, stride = 2, batch = 7;

            VariableField x = new Tensor(Shape.Map3D(inchannels, inwidth, inheight, indepth, batch));

            Layer layer = new ComplexDeconvolution3D(inchannels, outchannels, kwidth, kheight, kdepth, stride, use_bias:true, pad_mode:PaddingMode.Edge, "conv");

            Field y = layer.Forward(x);

            (Flow flow, Parameters parameters) = Flow.Optimize(y);
            flow.Execute();

            Assert.AreEqual(2, parameters.Count);
            Assert.AreEqual(inchannels, layer.InChannels);
            Assert.AreEqual(outchannels, layer.OutChannels);
            Assert.AreEqual(kwidth, layer.Width);
            Assert.AreEqual(kheight, layer.Height);
            Assert.AreEqual(kdepth, layer.Depth);
        }
    }
}
