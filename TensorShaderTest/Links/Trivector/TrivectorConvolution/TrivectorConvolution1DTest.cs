using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.TrivectorConvolution {
    [TestClass]
    public class TrivectorConvolution1DTest {
        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 9, outchannels = 12, kwidth = 3, stride = 2, inwidth = 7;
            int outwidth = (inwidth - kwidth) / stride + 1, batch = 3;

            float[] xval = (new float[inwidth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[outwidth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * outchannels * inchannels / 9 * 4]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Tensor xtensor = new Tensor(Shape.Map1D(inchannels, inwidth, batch), xval);
            Tensor ytensor = new Tensor(Shape.Map1D(outchannels, outwidth, batch), yval);
            Tensor wtensor = new Tensor(Shape.Kernel1D(inchannels / 3 * 4, outchannels / 3, kwidth), wval);

            ParameterField x = xtensor;
            ParameterField w = wtensor;
            VariableField y_actual = ytensor;

            Field y_expect = TrivectorConvolution1D(x, w, stride);
            Field err = y_expect - y_actual;

            (Flow flow, Parameters Parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradTensor.State;
            float[] gw_actual = w.GradTensor.State;

            AssertError.Tolerance(gw_expect, gw_actual, 1e-6f, 1e-5f, $"not equal gw");

            AssertError.Tolerance(gx_expect, gx_actual, 1e-6f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = new float[] {
            -8.049368383e-04f,  -9.929351354e-04f,  -3.970793001e-04f,  -7.478715623e-04f,  -9.241621379e-04f,  -3.651065321e-04f,
            -6.929044060e-04f,  -8.578685373e-04f,  -3.344324214e-04f,  -2.586294248e-04f,  -3.312993584e-04f,  -9.911746461e-05f,
            -2.267415850e-04f,  -2.922791232e-04f,  -8.272858397e-05f,  -1.969518648e-04f,  -2.557382850e-04f,  -6.763836061e-05f,
            -2.648181764e-03f,  -2.848869333e-03f,  -2.134487158e-03f,  -2.467619902e-03f,  -2.654055122e-03f,  -1.988589176e-03f,
            -2.295153427e-03f,  -2.468097857e-03f,  -1.849115992e-03f,  -9.433327006e-04f,  -1.016530029e-03f,  -7.435343764e-04f,
            -8.414484575e-04f,  -9.075138883e-04f,  -6.599550904e-04f,  -7.455614824e-04f,  -8.048752974e-04f,  -5.815019464e-04f,
            -4.579065669e-03f,  -4.780664700e-03f,  -3.974478775e-03f,  -4.258511000e-03f,  -4.445858678e-03f,  -3.694199981e-03f,
            -3.953850016e-03f,  -4.127705908e-03f,  -3.428000956e-03f,  -1.628035976e-03f,  -1.701760700e-03f,  -1.387951288e-03f,
            -1.456155330e-03f,  -1.522748653e-03f,  -1.237181597e-03f,  -1.294171100e-03f,  -1.354012310e-03f,  -1.095365532e-03f,
            -2.186316940e-04f,  -2.318125288e-04f,  -1.696543488e-04f,  -1.655080428e-04f,  -1.761089160e-04f,  -1.263281769e-04f,
            -1.222808078e-04f,  -1.306810059e-04f,  -9.195563158e-05f,  -5.712482720e-03f,  -5.901746273e-03f,  -5.061630833e-03f,
            -5.340361805e-03f,  -5.517961182e-03f,  -4.726281290e-03f,  -4.980914028e-03f,  -5.147227023e-03f,  -4.402554600e-03f,
            -2.083458890e-03f,  -2.157686705e-03f,  -1.804544656e-03f,  -1.863415639e-03f,  -1.930512801e-03f,  -1.608669357e-03f,
            -1.656045526e-03f,  -1.716389829e-03f,  -1.424416911e-03f,  -7.820643432e-03f,  -8.023945119e-03f,  -7.032025816e-03f,
            -7.263770747e-03f,  -7.452823569e-03f,  -6.527154996e-03f,  -6.736143488e-03f,  -6.911702036e-03f,  -6.049357367e-03f,
            -2.768162166e-03f,  -2.842917376e-03f,  -2.448961568e-03f,  -2.478122511e-03f,  -2.545747566e-03f,  -2.185895863e-03f,
            -2.204655144e-03f,  -2.265526841e-03f,  -1.938280497e-03f,  -9.751527337e-03f,  -9.955740486e-03f,  -8.872017432e-03f,
            -9.054661845e-03f,  -9.244627125e-03f,  -8.232765801e-03f,  -8.394840076e-03f,  -8.571310087e-03f,  -7.628242331e-03f,
            -3.452865441e-03f,  -3.528148046e-03f,  -3.093378480e-03f,  -3.092829384e-03f,  -3.160982331e-03f,  -2.763122370e-03f,
            -2.753264762e-03f,  -2.814663853e-03f,  -2.452144082e-03f,  -4.835474792e-04f,  -4.980771770e-04f,  -4.026414736e-04f,
            -3.691686449e-04f,  -3.810783191e-04f,  -3.037192392e-04f,  -2.752612458e-04f,  -2.849266994e-04f,  -2.240748276e-04f,
            -1.062002860e-02f,  -1.081055741e-02f,  -9.726182366e-03f,  -9.932852048e-03f,  -1.011176023e-02f,  -9.087456047e-03f,
            -9.268923651e-03f,  -9.436585508e-03f,  -8.470676779e-03f,  -3.908288355e-03f,  -3.984074052e-03f,  -3.509971847e-03f,
            -3.500089693e-03f,  -3.568746478e-03f,  -3.134610129e-03f,  -3.115139188e-03f,  -3.177041372e-03f,  -2.781195461e-03f,
            -1.299310510e-02f,  -1.319902090e-02f,  -1.192956447e-02f,  -1.205992159e-02f,  -1.225159202e-02f,  -1.106572082e-02f,
            -1.117713355e-02f,  -1.135530622e-02f,  -1.024959874e-02f,  -4.592991631e-03f,  -4.669304723e-03f,  -4.154388759e-03f,
            -4.114796565e-03f,  -4.183981243e-03f,  -3.711836636e-03f,  -3.663748806e-03f,  -3.726178385e-03f,  -3.295059047e-03f,
            -1.492398900e-02f,  -1.513081627e-02f,  -1.376955609e-02f,  -1.385081269e-02f,  -1.404339557e-02f,  -1.277133162e-02f,
            -1.283583014e-02f,  -1.301491427e-02f,  -1.182848371e-02f,  -5.277694907e-03f,  -5.354535393e-03f,  -4.798805671e-03f,
            -4.729503438e-03f,  -4.799216008e-03f,  -4.289063142e-03f,  -4.212358423e-03f,  -4.275315397e-03f,  -3.808922632e-03f,
            -7.484632644e-04f,  -7.643418252e-04f,  -6.356285985e-04f,  -5.728292470e-04f,  -5.860477223e-04f,  -4.811103014e-04f,
            -4.282416839e-04f,  -4.391723928e-04f,  -3.561940237e-04f
        };

        float[] gw_expect = new float[] {
            -1.694188566e-02f,  -1.517272459e-02f,  -1.651116385e-02f,  -1.838389373e-02f,  -1.688593101e-02f,  -1.510978958e-02f,
            -1.644452741e-02f,  -1.832845236e-02f,  -1.680591070e-02f,  -1.502500917e-02f,  -1.635437495e-02f,  -1.824617546e-02f,
            -2.250678354e-02f,  -2.072761693e-02f,  -2.200237337e-02f,  -2.371529830e-02f,  -2.239740090e-02f,  -2.061260299e-02f,
            -2.187855270e-02f,  -2.359951937e-02f,  -2.225160775e-02f,  -2.046316887e-02f,  -2.171893458e-02f,  -2.344472652e-02f,
            -2.608743284e-02f,  -2.431146505e-02f,  -2.551751426e-02f,  -2.707656590e-02f,  -2.589310218e-02f,  -2.411234714e-02f,
            -2.530563987e-02f,  -2.686973446e-02f,  -2.565127920e-02f,  -2.386751524e-02f,  -2.504694074e-02f,  -2.661296480e-02f,
            -2.789688950e-02f,  -2.613700239e-02f,  -2.726980368e-02f,  -2.868091368e-02f,  -2.759177458e-02f,  -2.582751989e-02f,
            -2.694464961e-02f,  -2.835795829e-02f,  -2.722934862e-02f,  -2.546231060e-02f,  -2.656289761e-02f,  -2.797539450e-02f,
            -1.211101992e-02f,  -1.071814693e-02f,  -1.170377895e-02f,  -1.311177103e-02f,  -1.187457287e-02f,  -1.049137143e-02f,
            -1.146077232e-02f,  -1.285506317e-02f,  -1.161406017e-02f,  -1.024275053e-02f,  -1.119424967e-02f,  -1.257151978e-02f,
            -1.539204020e-02f,  -1.398091455e-02f,  -1.489624580e-02f,  -1.615415081e-02f,  -1.500957867e-02f,  -1.360774922e-02f,
            -1.450394432e-02f,  -1.574576741e-02f,  -1.459070662e-02f,  -1.320016370e-02f,  -1.407584541e-02f,  -1.529837007e-02f,
            -1.682814836e-02f,  -1.541043252e-02f,  -1.625394562e-02f,  -1.736815880e-02f,  -1.627762534e-02f,  -1.486845971e-02f,
            -1.569088567e-02f,  -1.678679074e-02f,  -1.567961001e-02f,  -1.428077292e-02f,  -1.508100098e-02f,  -1.615548447e-02f,
            -1.664945181e-02f,  -1.523672762e-02f,  -1.600702612e-02f,  -1.698394270e-02f,  -1.591450410e-02f,  -1.450929413e-02f,
            -1.525738758e-02f,  -1.621392440e-02f,  -1.512224536e-02f,  -1.372613384e-02f,  -1.445115111e-02f,  -1.538429772e-02f,
            -6.413790697e-03f,  -5.477134918e-03f,  -6.049817137e-03f,  -6.873569169e-03f,  -5.996851260e-03f,  -5.086518930e-03f,
            -5.630440320e-03f,  -6.415594817e-03f,  -5.555846170e-03f,  -4.674057543e-03f,  -5.187547477e-03f,  -5.930784932e-03f,
            -6.966518179e-03f,  -5.995085474e-03f,  -6.501410383e-03f,  -7.188501828e-03f,  -6.310977762e-03f,  -5.363768751e-03f,
            -5.840628111e-03f,  -6.487513944e-03f,  -5.619026827e-03f,  -4.698031842e-03f,  -5.144048398e-03f,  -5.747512129e-03f,
            -5.859140578e-03f,  -4.863696497e-03f,  -5.304686299e-03f,  -5.861975961e-03f,  -4.952425203e-03f,  -3.978868795e-03f,
            -4.390440794e-03f,  -4.906071293e-03f,  -3.998217515e-03f,  -3.048327108e-03f,  -3.429370549e-03f,  -3.900228410e-03f,
            -3.338816748e-03f,  -2.330288085e-03f,  -2.706723116e-03f,  -3.141069802e-03f,  -2.174036257e-03f,  -1.184903602e-03f,
            -1.532600113e-03f,  -1.923988607e-03f,  -9.519447283e-04f,  1.620768038e-05f,  -3.018791808e-04f,  -6.472990267e-04f,
        };
    }
}
