using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.Connection3D {
    [TestClass]
    public class PointwisewiseDeconvolutionTest {
        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 2, outchannels = 3, width = 7, height = 6, depth = 5, batch = 2;

            float[] xval = (new float[width * height * depth * inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[width * height * depth * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[outchannels * inchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            ParameterField y = new Tensor(Shape.Map3D(outchannels, width, height, depth, batch), yval);
            ParameterField w = new Tensor(Shape.Kernel0D(inchannels, outchannels), wval);
            VariableField x_actual = new Tensor(Shape.Map3D(inchannels, width, height, depth, batch), xval);

            Field x_expect = PointwiseDeconvolution3D(y, w);
            Field err = Abs(x_expect - x_actual);

            (Flow flow, Parameters parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gy_actual = y.GradState;
            float[] gw_actual = w.GradState;

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"not equal gw");

            AssertError.Tolerance(gy_expect, gy_actual, 1e-7f, 1e-5f, $"not equal gy");
        }

        float[] gy_expect = new float[] {
            -3.967000000e-06f,  -1.981000000e-06f,  5.000000000e-09f,  -2.176000000e-05f,  -1.186400000e-05f,  -1.968000000e-06f,
            -3.955300000e-05f,  -2.174700000e-05f,  -3.941000000e-06f,  -5.734600000e-05f,  -3.163000000e-05f,  -5.914000000e-06f,
            -7.513900000e-05f,  -4.151300000e-05f,  -7.887000000e-06f,  -9.293200000e-05f,  -5.139600000e-05f,  -9.860000000e-06f,
            -1.107250000e-04f,  -6.127900000e-05f,  -1.183300000e-05f,  -1.285180000e-04f,  -7.116200000e-05f,  -1.380600000e-05f,
            -1.463110000e-04f,  -8.104500000e-05f,  -1.577900000e-05f,  -1.641040000e-04f,  -9.092800000e-05f,  -1.775200000e-05f,
            -1.818970000e-04f,  -1.008110000e-04f,  -1.972500000e-05f,  -1.996900000e-04f,  -1.106940000e-04f,  -2.169800000e-05f,
            -2.174830000e-04f,  -1.205770000e-04f,  -2.367100000e-05f,  -2.352760000e-04f,  -1.304600000e-04f,  -2.564400000e-05f,
            -2.530690000e-04f,  -1.403430000e-04f,  -2.761700000e-05f,  -2.708620000e-04f,  -1.502260000e-04f,  -2.959000000e-05f,
            -2.886550000e-04f,  -1.601090000e-04f,  -3.156300000e-05f,  -3.064480000e-04f,  -1.699920000e-04f,  -3.353600000e-05f,
            -3.242410000e-04f,  -1.798750000e-04f,  -3.550900000e-05f,  -3.420340000e-04f,  -1.897580000e-04f,  -3.748200000e-05f,
            -3.598270000e-04f,  -1.996410000e-04f,  -3.945500000e-05f,  -3.776200000e-04f,  -2.095240000e-04f,  -4.142800000e-05f,
            -3.954130000e-04f,  -2.194070000e-04f,  -4.340100000e-05f,  -4.132060000e-04f,  -2.292900000e-04f,  -4.537400000e-05f,
            -4.309990000e-04f,  -2.391730000e-04f,  -4.734700000e-05f,  -4.487920000e-04f,  -2.490560000e-04f,  -4.932000000e-05f,
            -4.665850000e-04f,  -2.589390000e-04f,  -5.129300000e-05f,  -4.843780000e-04f,  -2.688220000e-04f,  -5.326600000e-05f,
            -5.021710000e-04f,  -2.787050000e-04f,  -5.523900000e-05f,  -5.199640000e-04f,  -2.885880000e-04f,  -5.721200000e-05f,
            -5.377570000e-04f,  -2.984710000e-04f,  -5.918500000e-05f,  -5.555500000e-04f,  -3.083540000e-04f,  -6.115800000e-05f,
            -5.733430000e-04f,  -3.182370000e-04f,  -6.313100000e-05f,  -5.911360000e-04f,  -3.281200000e-04f,  -6.510400000e-05f,
            -6.089290000e-04f,  -3.380030000e-04f,  -6.707700000e-05f,  -6.267220000e-04f,  -3.478860000e-04f,  -6.905000000e-05f,
            -6.445150000e-04f,  -3.577690000e-04f,  -7.102300000e-05f,  -6.623080000e-04f,  -3.676520000e-04f,  -7.299600000e-05f,
            -6.801010000e-04f,  -3.775350000e-04f,  -7.496900000e-05f,  -6.978940000e-04f,  -3.874180000e-04f,  -7.694200000e-05f,
            -7.156870000e-04f,  -3.973010000e-04f,  -7.891500000e-05f,  -7.334800000e-04f,  -4.071840000e-04f,  -8.088800000e-05f,
            -7.512730000e-04f,  -4.170670000e-04f,  -8.286100000e-05f,  -7.690660000e-04f,  -4.269500000e-04f,  -8.483400000e-05f,
            -7.868590000e-04f,  -4.368330000e-04f,  -8.680700000e-05f,  -8.046520000e-04f,  -4.467160000e-04f,  -8.878000000e-05f,
            -8.224450000e-04f,  -4.565990000e-04f,  -9.075300000e-05f,  -8.402380000e-04f,  -4.664820000e-04f,  -9.272600000e-05f,
            -8.580310000e-04f,  -4.763650000e-04f,  -9.469900000e-05f,  -8.758240000e-04f,  -4.862480000e-04f,  -9.667200000e-05f,
            -8.936170000e-04f,  -4.961310000e-04f,  -9.864500000e-05f,  -9.114100000e-04f,  -5.060140000e-04f,  -1.006180000e-04f,
            -9.292030000e-04f,  -5.158970000e-04f,  -1.025910000e-04f,  -9.469960000e-04f,  -5.257800000e-04f,  -1.045640000e-04f,
            -9.647890000e-04f,  -5.356630000e-04f,  -1.065370000e-04f,  -9.825820000e-04f,  -5.455460000e-04f,  -1.085100000e-04f,
            -1.000375000e-03f,  -5.554290000e-04f,  -1.104830000e-04f,  -1.018168000e-03f,  -5.653120000e-04f,  -1.124560000e-04f,
            -1.035961000e-03f,  -5.751950000e-04f,  -1.144290000e-04f,  -1.053754000e-03f,  -5.850780000e-04f,  -1.164020000e-04f,
            -1.071547000e-03f,  -5.949610000e-04f,  -1.183750000e-04f,  -1.089340000e-03f,  -6.048440000e-04f,  -1.203480000e-04f,
            -1.107133000e-03f,  -6.147270000e-04f,  -1.223210000e-04f,  -1.124926000e-03f,  -6.246100000e-04f,  -1.242940000e-04f,
            -1.142719000e-03f,  -6.344930000e-04f,  -1.262670000e-04f,  -1.160512000e-03f,  -6.443760000e-04f,  -1.282400000e-04f,
            -1.178305000e-03f,  -6.542590000e-04f,  -1.302130000e-04f,  -1.196098000e-03f,  -6.641420000e-04f,  -1.321860000e-04f,
            -1.213891000e-03f,  -6.740250000e-04f,  -1.341590000e-04f,  -1.231684000e-03f,  -6.839080000e-04f,  -1.361320000e-04f,
            -1.249477000e-03f,  -6.937910000e-04f,  -1.381050000e-04f,  -1.267270000e-03f,  -7.036740000e-04f,  -1.400780000e-04f,
            -1.285063000e-03f,  -7.135570000e-04f,  -1.420510000e-04f,  -1.302856000e-03f,  -7.234400000e-04f,  -1.440240000e-04f,
            -1.320649000e-03f,  -7.333230000e-04f,  -1.459970000e-04f,  -1.338442000e-03f,  -7.432060000e-04f,  -1.479700000e-04f,
            -1.356235000e-03f,  -7.530890000e-04f,  -1.499430000e-04f,  -1.374028000e-03f,  -7.629720000e-04f,  -1.519160000e-04f,
            -1.391821000e-03f,  -7.728550000e-04f,  -1.538890000e-04f,  -1.409614000e-03f,  -7.827380000e-04f,  -1.558620000e-04f,
            -1.427407000e-03f,  -7.926210000e-04f,  -1.578350000e-04f,  -1.445200000e-03f,  -8.025040000e-04f,  -1.598080000e-04f,
            -1.462993000e-03f,  -8.123870000e-04f,  -1.617810000e-04f,  -1.480786000e-03f,  -8.222700000e-04f,  -1.637540000e-04f,
            -1.498579000e-03f,  -8.321530000e-04f,  -1.657270000e-04f,  -1.516372000e-03f,  -8.420360000e-04f,  -1.677000000e-04f,
            -1.534165000e-03f,  -8.519190000e-04f,  -1.696730000e-04f,  -1.551958000e-03f,  -8.618020000e-04f,  -1.716460000e-04f,
            -1.569751000e-03f,  -8.716850000e-04f,  -1.736190000e-04f,  -1.587544000e-03f,  -8.815680000e-04f,  -1.755920000e-04f,
            -1.605337000e-03f,  -8.914510000e-04f,  -1.775650000e-04f,  -1.623130000e-03f,  -9.013340000e-04f,  -1.795380000e-04f,
            -1.640923000e-03f,  -9.112170000e-04f,  -1.815110000e-04f,  -1.658716000e-03f,  -9.211000000e-04f,  -1.834840000e-04f,
            -1.676509000e-03f,  -9.309830000e-04f,  -1.854570000e-04f,  -1.694302000e-03f,  -9.408660000e-04f,  -1.874300000e-04f,
            -1.712095000e-03f,  -9.507490000e-04f,  -1.894030000e-04f,  -1.729888000e-03f,  -9.606320000e-04f,  -1.913760000e-04f,
            -1.747681000e-03f,  -9.705150000e-04f,  -1.933490000e-04f,  -1.765474000e-03f,  -9.803980000e-04f,  -1.953220000e-04f,
            -1.783267000e-03f,  -9.902810000e-04f,  -1.972950000e-04f,  -1.801060000e-03f,  -1.000164000e-03f,  -1.992680000e-04f,
            -1.818853000e-03f,  -1.010047000e-03f,  -2.012410000e-04f,  -1.836646000e-03f,  -1.019930000e-03f,  -2.032140000e-04f,
            -1.854439000e-03f,  -1.029813000e-03f,  -2.051870000e-04f,  -1.872232000e-03f,  -1.039696000e-03f,  -2.071600000e-04f,
            -1.890025000e-03f,  -1.049579000e-03f,  -2.091330000e-04f,  -1.907818000e-03f,  -1.059462000e-03f,  -2.111060000e-04f,
            -1.925611000e-03f,  -1.069345000e-03f,  -2.130790000e-04f,  -1.943404000e-03f,  -1.079228000e-03f,  -2.150520000e-04f,
            -1.961197000e-03f,  -1.089111000e-03f,  -2.170250000e-04f,  -1.978990000e-03f,  -1.098994000e-03f,  -2.189980000e-04f,
            -1.996783000e-03f,  -1.108877000e-03f,  -2.209710000e-04f,  -2.014576000e-03f,  -1.118760000e-03f,  -2.229440000e-04f,
            -2.032369000e-03f,  -1.128643000e-03f,  -2.249170000e-04f,  -2.050162000e-03f,  -1.138526000e-03f,  -2.268900000e-04f,
            -2.067955000e-03f,  -1.148409000e-03f,  -2.288630000e-04f,  -2.085748000e-03f,  -1.158292000e-03f,  -2.308360000e-04f,
            -2.103541000e-03f,  -1.168175000e-03f,  -2.328090000e-04f,  -2.121334000e-03f,  -1.178058000e-03f,  -2.347820000e-04f,
            -2.139127000e-03f,  -1.187941000e-03f,  -2.367550000e-04f,  -2.156920000e-03f,  -1.197824000e-03f,  -2.387280000e-04f,
            -2.174713000e-03f,  -1.207707000e-03f,  -2.407010000e-04f,  -2.192506000e-03f,  -1.217590000e-03f,  -2.426740000e-04f,
            -2.210299000e-03f,  -1.227473000e-03f,  -2.446470000e-04f,  -2.228092000e-03f,  -1.237356000e-03f,  -2.466200000e-04f,
            -2.245885000e-03f,  -1.247239000e-03f,  -2.485930000e-04f,  -2.263678000e-03f,  -1.257122000e-03f,  -2.505660000e-04f,
            -2.281471000e-03f,  -1.267005000e-03f,  -2.525390000e-04f,  -2.299264000e-03f,  -1.276888000e-03f,  -2.545120000e-04f,
            -2.317057000e-03f,  -1.286771000e-03f,  -2.564850000e-04f,  -2.334850000e-03f,  -1.296654000e-03f,  -2.584580000e-04f,
            -2.352643000e-03f,  -1.306537000e-03f,  -2.604310000e-04f,  -2.370436000e-03f,  -1.316420000e-03f,  -2.624040000e-04f,
            -2.388229000e-03f,  -1.326303000e-03f,  -2.643770000e-04f,  -2.406022000e-03f,  -1.336186000e-03f,  -2.663500000e-04f,
            -2.423815000e-03f,  -1.346069000e-03f,  -2.683230000e-04f,  -2.441608000e-03f,  -1.355952000e-03f,  -2.702960000e-04f,
            -2.459401000e-03f,  -1.365835000e-03f,  -2.722690000e-04f,  -2.477194000e-03f,  -1.375718000e-03f,  -2.742420000e-04f,
            -2.494987000e-03f,  -1.385601000e-03f,  -2.762150000e-04f,  -2.512780000e-03f,  -1.395484000e-03f,  -2.781880000e-04f,
            -2.530573000e-03f,  -1.405367000e-03f,  -2.801610000e-04f,  -2.548366000e-03f,  -1.415250000e-03f,  -2.821340000e-04f,
            -2.566159000e-03f,  -1.425133000e-03f,  -2.841070000e-04f,  -2.583952000e-03f,  -1.435016000e-03f,  -2.860800000e-04f,
            -2.601745000e-03f,  -1.444899000e-03f,  -2.880530000e-04f,  -2.619538000e-03f,  -1.454782000e-03f,  -2.900260000e-04f,
            -2.637331000e-03f,  -1.464665000e-03f,  -2.919990000e-04f,  -2.655124000e-03f,  -1.474548000e-03f,  -2.939720000e-04f,
            -2.672917000e-03f,  -1.484431000e-03f,  -2.959450000e-04f,  -2.690710000e-03f,  -1.494314000e-03f,  -2.979180000e-04f,
            -2.708503000e-03f,  -1.504197000e-03f,  -2.998910000e-04f,  -2.726296000e-03f,  -1.514080000e-03f,  -3.018640000e-04f,
            -2.744089000e-03f,  -1.523963000e-03f,  -3.038370000e-04f,  -2.761882000e-03f,  -1.533846000e-03f,  -3.058100000e-04f,
            -2.779675000e-03f,  -1.543729000e-03f,  -3.077830000e-04f,  -2.797468000e-03f,  -1.553612000e-03f,  -3.097560000e-04f,
            -2.815261000e-03f,  -1.563495000e-03f,  -3.117290000e-04f,  -2.833054000e-03f,  -1.573378000e-03f,  -3.137020000e-04f,
            -2.850847000e-03f,  -1.583261000e-03f,  -3.156750000e-04f,  -2.868640000e-03f,  -1.593144000e-03f,  -3.176480000e-04f,
            -2.886433000e-03f,  -1.603027000e-03f,  -3.196210000e-04f,  -2.904226000e-03f,  -1.612910000e-03f,  -3.215940000e-04f,
            -2.922019000e-03f,  -1.622793000e-03f,  -3.235670000e-04f,  -2.939812000e-03f,  -1.632676000e-03f,  -3.255400000e-04f,
            -2.957605000e-03f,  -1.642559000e-03f,  -3.275130000e-04f,  -2.975398000e-03f,  -1.652442000e-03f,  -3.294860000e-04f,
            -2.993191000e-03f,  -1.662325000e-03f,  -3.314590000e-04f,  -3.010984000e-03f,  -1.672208000e-03f,  -3.334320000e-04f,
            -3.028777000e-03f,  -1.682091000e-03f,  -3.354050000e-04f,  -3.046570000e-03f,  -1.691974000e-03f,  -3.373780000e-04f,
            -3.064363000e-03f,  -1.701857000e-03f,  -3.393510000e-04f,  -3.082156000e-03f,  -1.711740000e-03f,  -3.413240000e-04f,
            -3.099949000e-03f,  -1.721623000e-03f,  -3.432970000e-04f,  -3.117742000e-03f,  -1.731506000e-03f,  -3.452700000e-04f,
            -3.135535000e-03f,  -1.741389000e-03f,  -3.472430000e-04f,  -3.153328000e-03f,  -1.751272000e-03f,  -3.492160000e-04f,
            -3.171121000e-03f,  -1.761155000e-03f,  -3.511890000e-04f,  -3.188914000e-03f,  -1.771038000e-03f,  -3.531620000e-04f,
            -3.206707000e-03f,  -1.780921000e-03f,  -3.551350000e-04f,  -3.224500000e-03f,  -1.790804000e-03f,  -3.571080000e-04f,
            -3.242293000e-03f,  -1.800687000e-03f,  -3.590810000e-04f,  -3.260086000e-03f,  -1.810570000e-03f,  -3.610540000e-04f,
            -3.277879000e-03f,  -1.820453000e-03f,  -3.630270000e-04f,  -3.295672000e-03f,  -1.830336000e-03f,  -3.650000000e-04f,
            -3.313465000e-03f,  -1.840219000e-03f,  -3.669730000e-04f,  -3.331258000e-03f,  -1.850102000e-03f,  -3.689460000e-04f,
            -3.349051000e-03f,  -1.859985000e-03f,  -3.709190000e-04f,  -3.366844000e-03f,  -1.869868000e-03f,  -3.728920000e-04f,
            -3.384637000e-03f,  -1.879751000e-03f,  -3.748650000e-04f,  -3.402430000e-03f,  -1.889634000e-03f,  -3.768380000e-04f,
            -3.420223000e-03f,  -1.899517000e-03f,  -3.788110000e-04f,  -3.438016000e-03f,  -1.909400000e-03f,  -3.807840000e-04f,
            -3.455809000e-03f,  -1.919283000e-03f,  -3.827570000e-04f,  -3.473602000e-03f,  -1.929166000e-03f,  -3.847300000e-04f,
            -3.491395000e-03f,  -1.939049000e-03f,  -3.867030000e-04f,  -3.509188000e-03f,  -1.948932000e-03f,  -3.886760000e-04f,
            -3.526981000e-03f,  -1.958815000e-03f,  -3.906490000e-04f,  -3.544774000e-03f,  -1.968698000e-03f,  -3.926220000e-04f,
            -3.562567000e-03f,  -1.978581000e-03f,  -3.945950000e-04f,  -3.580360000e-03f,  -1.988464000e-03f,  -3.965680000e-04f,
            -3.598153000e-03f,  -1.998347000e-03f,  -3.985410000e-04f,  -3.615946000e-03f,  -2.008230000e-03f,  -4.005140000e-04f,
            -3.633739000e-03f,  -2.018113000e-03f,  -4.024870000e-04f,  -3.651532000e-03f,  -2.027996000e-03f,  -4.044600000e-04f,
            -3.669325000e-03f,  -2.037879000e-03f,  -4.064330000e-04f,  -3.687118000e-03f,  -2.047762000e-03f,  -4.084060000e-04f,
            -3.704911000e-03f,  -2.057645000e-03f,  -4.103790000e-04f,  -3.722704000e-03f,  -2.067528000e-03f,  -4.123520000e-04f,
            -3.740497000e-03f,  -2.077411000e-03f,  -4.143250000e-04f,  -3.758290000e-03f,  -2.087294000e-03f,  -4.162980000e-04f,
            -3.776083000e-03f,  -2.097177000e-03f,  -4.182710000e-04f,  -3.793876000e-03f,  -2.107060000e-03f,  -4.202440000e-04f,
            -3.811669000e-03f,  -2.116943000e-03f,  -4.222170000e-04f,  -3.829462000e-03f,  -2.126826000e-03f,  -4.241900000e-04f,
            -3.847255000e-03f,  -2.136709000e-03f,  -4.261630000e-04f,  -3.865048000e-03f,  -2.146592000e-03f,  -4.281360000e-04f,
            -3.882841000e-03f,  -2.156475000e-03f,  -4.301090000e-04f,  -3.900634000e-03f,  -2.166358000e-03f,  -4.320820000e-04f,
            -3.918427000e-03f,  -2.176241000e-03f,  -4.340550000e-04f,  -3.936220000e-03f,  -2.186124000e-03f,  -4.360280000e-04f,
            -3.954013000e-03f,  -2.196007000e-03f,  -4.380010000e-04f,  -3.971806000e-03f,  -2.205890000e-03f,  -4.399740000e-04f,
            -3.989599000e-03f,  -2.215773000e-03f,  -4.419470000e-04f,  -4.007392000e-03f,  -2.225656000e-03f,  -4.439200000e-04f,
            -4.025185000e-03f,  -2.235539000e-03f,  -4.458930000e-04f,  -4.042978000e-03f,  -2.245422000e-03f,  -4.478660000e-04f,
            -4.060771000e-03f,  -2.255305000e-03f,  -4.498390000e-04f,  -4.078564000e-03f,  -2.265188000e-03f,  -4.518120000e-04f,
            -4.096357000e-03f,  -2.275071000e-03f,  -4.537850000e-04f,  -4.114150000e-03f,  -2.284954000e-03f,  -4.557580000e-04f,
            -4.131943000e-03f,  -2.294837000e-03f,  -4.577310000e-04f,  -4.149736000e-03f,  -2.304720000e-03f,  -4.597040000e-04f,
            -4.167529000e-03f,  -2.314603000e-03f,  -4.616770000e-04f,  -4.185322000e-03f,  -2.324486000e-03f,  -4.636500000e-04f,
            -4.203115000e-03f,  -2.334369000e-03f,  -4.656230000e-04f,  -4.220908000e-03f,  -2.344252000e-03f,  -4.675960000e-04f,
            -4.238701000e-03f,  -2.354135000e-03f,  -4.695690000e-04f,  -4.256494000e-03f,  -2.364018000e-03f,  -4.715420000e-04f,
            -4.274287000e-03f,  -2.373901000e-03f,  -4.735150000e-04f,  -4.292080000e-03f,  -2.383784000e-03f,  -4.754880000e-04f,
            -4.309873000e-03f,  -2.393667000e-03f,  -4.774610000e-04f,  -4.327666000e-03f,  -2.403550000e-03f,  -4.794340000e-04f,
            -4.345459000e-03f,  -2.413433000e-03f,  -4.814070000e-04f,  -4.363252000e-03f,  -2.423316000e-03f,  -4.833800000e-04f,
            -4.381045000e-03f,  -2.433199000e-03f,  -4.853530000e-04f,  -4.398838000e-03f,  -2.443082000e-03f,  -4.873260000e-04f,
            -4.416631000e-03f,  -2.452965000e-03f,  -4.892990000e-04f,  -4.434424000e-03f,  -2.462848000e-03f,  -4.912720000e-04f,
            -4.452217000e-03f,  -2.472731000e-03f,  -4.932450000e-04f,  -4.470010000e-03f,  -2.482614000e-03f,  -4.952180000e-04f,
            -4.487803000e-03f,  -2.492497000e-03f,  -4.971910000e-04f,  -4.505596000e-03f,  -2.502380000e-03f,  -4.991640000e-04f,
            -4.523389000e-03f,  -2.512263000e-03f,  -5.011370000e-04f,  -4.541182000e-03f,  -2.522146000e-03f,  -5.031100000e-04f,
            -4.558975000e-03f,  -2.532029000e-03f,  -5.050830000e-04f,  -4.576768000e-03f,  -2.541912000e-03f,  -5.070560000e-04f,
            -4.594561000e-03f,  -2.551795000e-03f,  -5.090290000e-04f,  -4.612354000e-03f,  -2.561678000e-03f,  -5.110020000e-04f,
            -4.630147000e-03f,  -2.571561000e-03f,  -5.129750000e-04f,  -4.647940000e-03f,  -2.581444000e-03f,  -5.149480000e-04f,
            -4.665733000e-03f,  -2.591327000e-03f,  -5.169210000e-04f,  -4.683526000e-03f,  -2.601210000e-03f,  -5.188940000e-04f,
            -4.701319000e-03f,  -2.611093000e-03f,  -5.208670000e-04f,  -4.719112000e-03f,  -2.620976000e-03f,  -5.228400000e-04f,
            -4.736905000e-03f,  -2.630859000e-03f,  -5.248130000e-04f,  -4.754698000e-03f,  -2.640742000e-03f,  -5.267860000e-04f,
            -4.772491000e-03f,  -2.650625000e-03f,  -5.287590000e-04f,  -4.790284000e-03f,  -2.660508000e-03f,  -5.307320000e-04f,
            -4.808077000e-03f,  -2.670391000e-03f,  -5.327050000e-04f,  -4.825870000e-03f,  -2.680274000e-03f,  -5.346780000e-04f,
            -4.843663000e-03f,  -2.690157000e-03f,  -5.366510000e-04f,  -4.861456000e-03f,  -2.700040000e-03f,  -5.386240000e-04f,
            -4.879249000e-03f,  -2.709923000e-03f,  -5.405970000e-04f,  -4.897042000e-03f,  -2.719806000e-03f,  -5.425700000e-04f,
            -4.914835000e-03f,  -2.729689000e-03f,  -5.445430000e-04f,  -4.932628000e-03f,  -2.739572000e-03f,  -5.465160000e-04f,
            -4.950421000e-03f,  -2.749455000e-03f,  -5.484890000e-04f,  -4.968214000e-03f,  -2.759338000e-03f,  -5.504620000e-04f,
            -4.986007000e-03f,  -2.769221000e-03f,  -5.524350000e-04f,  -5.003800000e-03f,  -2.779104000e-03f,  -5.544080000e-04f,
            -5.021593000e-03f,  -2.788987000e-03f,  -5.563810000e-04f,  -5.039386000e-03f,  -2.798870000e-03f,  -5.583540000e-04f,
            -5.057179000e-03f,  -2.808753000e-03f,  -5.603270000e-04f,  -5.074972000e-03f,  -2.818636000e-03f,  -5.623000000e-04f,
            -5.092765000e-03f,  -2.828519000e-03f,  -5.642730000e-04f,  -5.110558000e-03f,  -2.838402000e-03f,  -5.662460000e-04f,
            -5.128351000e-03f,  -2.848285000e-03f,  -5.682190000e-04f,  -5.146144000e-03f,  -2.858168000e-03f,  -5.701920000e-04f,
            -5.163937000e-03f,  -2.868051000e-03f,  -5.721650000e-04f,  -5.181730000e-03f,  -2.877934000e-03f,  -5.741380000e-04f,
            -5.199523000e-03f,  -2.887817000e-03f,  -5.761110000e-04f,  -5.217316000e-03f,  -2.897700000e-03f,  -5.780840000e-04f,
            -5.235109000e-03f,  -2.907583000e-03f,  -5.800570000e-04f,  -5.252902000e-03f,  -2.917466000e-03f,  -5.820300000e-04f,
            -5.270695000e-03f,  -2.927349000e-03f,  -5.840030000e-04f,  -5.288488000e-03f,  -2.937232000e-03f,  -5.859760000e-04f,
            -5.306281000e-03f,  -2.947115000e-03f,  -5.879490000e-04f,  -5.324074000e-03f,  -2.956998000e-03f,  -5.899220000e-04f,
            -5.341867000e-03f,  -2.966881000e-03f,  -5.918950000e-04f,  -5.359660000e-03f,  -2.976764000e-03f,  -5.938680000e-04f,
            -5.377453000e-03f,  -2.986647000e-03f,  -5.958410000e-04f,  -5.395246000e-03f,  -2.996530000e-03f,  -5.978140000e-04f,
            -5.413039000e-03f,  -3.006413000e-03f,  -5.997870000e-04f,  -5.430832000e-03f,  -3.016296000e-03f,  -6.017600000e-04f,
            -5.448625000e-03f,  -3.026179000e-03f,  -6.037330000e-04f,  -5.466418000e-03f,  -3.036062000e-03f,  -6.057060000e-04f,
            -5.484211000e-03f,  -3.045945000e-03f,  -6.076790000e-04f,  -5.502004000e-03f,  -3.055828000e-03f,  -6.096520000e-04f,
            -5.519797000e-03f,  -3.065711000e-03f,  -6.116250000e-04f,  -5.537590000e-03f,  -3.075594000e-03f,  -6.135980000e-04f,
            -5.555383000e-03f,  -3.085477000e-03f,  -6.155710000e-04f,  -5.573176000e-03f,  -3.095360000e-03f,  -6.175440000e-04f,
            -5.590969000e-03f,  -3.105243000e-03f,  -6.195170000e-04f,  -5.608762000e-03f,  -3.115126000e-03f,  -6.214900000e-04f,
            -5.626555000e-03f,  -3.125009000e-03f,  -6.234630000e-04f,  -5.644348000e-03f,  -3.134892000e-03f,  -6.254360000e-04f,
            -5.662141000e-03f,  -3.144775000e-03f,  -6.274090000e-04f,  -5.679934000e-03f,  -3.154658000e-03f,  -6.293820000e-04f,
            -5.697727000e-03f,  -3.164541000e-03f,  -6.313550000e-04f,  -5.715520000e-03f,  -3.174424000e-03f,  -6.333280000e-04f,
            -5.733313000e-03f,  -3.184307000e-03f,  -6.353010000e-04f,  -5.751106000e-03f,  -3.194190000e-03f,  -6.372740000e-04f,
            -5.768899000e-03f,  -3.204073000e-03f,  -6.392470000e-04f,  -5.786692000e-03f,  -3.213956000e-03f,  -6.412200000e-04f,
            -5.804485000e-03f,  -3.223839000e-03f,  -6.431930000e-04f,  -5.822278000e-03f,  -3.233722000e-03f,  -6.451660000e-04f,
            -5.840071000e-03f,  -3.243605000e-03f,  -6.471390000e-04f,  -5.857864000e-03f,  -3.253488000e-03f,  -6.491120000e-04f,
            -5.875657000e-03f,  -3.263371000e-03f,  -6.510850000e-04f,  -5.893450000e-03f,  -3.273254000e-03f,  -6.530580000e-04f,
            -5.911243000e-03f,  -3.283137000e-03f,  -6.550310000e-04f,  -5.929036000e-03f,  -3.293020000e-03f,  -6.570040000e-04f,
            -5.946829000e-03f,  -3.302903000e-03f,  -6.589770000e-04f,  -5.964622000e-03f,  -3.312786000e-03f,  -6.609500000e-04f,
            -5.982415000e-03f,  -3.322669000e-03f,  -6.629230000e-04f,  -6.000208000e-03f,  -3.332552000e-03f,  -6.648960000e-04f,
            -6.018001000e-03f,  -3.342435000e-03f,  -6.668690000e-04f,  -6.035794000e-03f,  -3.352318000e-03f,  -6.688420000e-04f,
            -6.053587000e-03f,  -3.362201000e-03f,  -6.708150000e-04f,  -6.071380000e-03f,  -3.372084000e-03f,  -6.727880000e-04f,
            -6.089173000e-03f,  -3.381967000e-03f,  -6.747610000e-04f,  -6.106966000e-03f,  -3.391850000e-03f,  -6.767340000e-04f,
            -6.124759000e-03f,  -3.401733000e-03f,  -6.787070000e-04f,  -6.142552000e-03f,  -3.411616000e-03f,  -6.806800000e-04f,
            -6.160345000e-03f,  -3.421499000e-03f,  -6.826530000e-04f,  -6.178138000e-03f,  -3.431382000e-03f,  -6.846260000e-04f,
            -6.195931000e-03f,  -3.441265000e-03f,  -6.865990000e-04f,  -6.213724000e-03f,  -3.451148000e-03f,  -6.885720000e-04f,
            -6.231517000e-03f,  -3.461031000e-03f,  -6.905450000e-04f,  -6.249310000e-03f,  -3.470914000e-03f,  -6.925180000e-04f,
            -6.267103000e-03f,  -3.480797000e-03f,  -6.944910000e-04f,  -6.284896000e-03f,  -3.490680000e-03f,  -6.964640000e-04f,
            -6.302689000e-03f,  -3.500563000e-03f,  -6.984370000e-04f,  -6.320482000e-03f,  -3.510446000e-03f,  -7.004100000e-04f,
            -6.338275000e-03f,  -3.520329000e-03f,  -7.023830000e-04f,  -6.356068000e-03f,  -3.530212000e-03f,  -7.043560000e-04f,
            -6.373861000e-03f,  -3.540095000e-03f,  -7.063290000e-04f,  -6.391654000e-03f,  -3.549978000e-03f,  -7.083020000e-04f,
            -6.409447000e-03f,  -3.559861000e-03f,  -7.102750000e-04f,  -6.427240000e-03f,  -3.569744000e-03f,  -7.122480000e-04f,
            -6.445033000e-03f,  -3.579627000e-03f,  -7.142210000e-04f,  -6.462826000e-03f,  -3.589510000e-03f,  -7.161940000e-04f,
            -6.480619000e-03f,  -3.599393000e-03f,  -7.181670000e-04f,  -6.498412000e-03f,  -3.609276000e-03f,  -7.201400000e-04f,
            -6.516205000e-03f,  -3.619159000e-03f,  -7.221130000e-04f,  -6.533998000e-03f,  -3.629042000e-03f,  -7.240860000e-04f,
            -6.551791000e-03f,  -3.638925000e-03f,  -7.260590000e-04f,  -6.569584000e-03f,  -3.648808000e-03f,  -7.280320000e-04f,
            -6.587377000e-03f,  -3.658691000e-03f,  -7.300050000e-04f,  -6.605170000e-03f,  -3.668574000e-03f,  -7.319780000e-04f,
            -6.622963000e-03f,  -3.678457000e-03f,  -7.339510000e-04f,  -6.640756000e-03f,  -3.688340000e-03f,  -7.359240000e-04f,
            -6.658549000e-03f,  -3.698223000e-03f,  -7.378970000e-04f,  -6.676342000e-03f,  -3.708106000e-03f,  -7.398700000e-04f,
            -6.694135000e-03f,  -3.717989000e-03f,  -7.418430000e-04f,  -6.711928000e-03f,  -3.727872000e-03f,  -7.438160000e-04f,
            -6.729721000e-03f,  -3.737755000e-03f,  -7.457890000e-04f,  -6.747514000e-03f,  -3.747638000e-03f,  -7.477620000e-04f,
            -6.765307000e-03f,  -3.757521000e-03f,  -7.497350000e-04f,  -6.783100000e-03f,  -3.767404000e-03f,  -7.517080000e-04f,
            -6.800893000e-03f,  -3.777287000e-03f,  -7.536810000e-04f,  -6.818686000e-03f,  -3.787170000e-03f,  -7.556540000e-04f,
            -6.836479000e-03f,  -3.797053000e-03f,  -7.576270000e-04f,  -6.854272000e-03f,  -3.806936000e-03f,  -7.596000000e-04f,
            -6.872065000e-03f,  -3.816819000e-03f,  -7.615730000e-04f,  -6.889858000e-03f,  -3.826702000e-03f,  -7.635460000e-04f,
            -6.907651000e-03f,  -3.836585000e-03f,  -7.655190000e-04f,  -6.925444000e-03f,  -3.846468000e-03f,  -7.674920000e-04f,
            -6.943237000e-03f,  -3.856351000e-03f,  -7.694650000e-04f,  -6.961030000e-03f,  -3.866234000e-03f,  -7.714380000e-04f,
            -6.978823000e-03f,  -3.876117000e-03f,  -7.734110000e-04f,  -6.996616000e-03f,  -3.886000000e-03f,  -7.753840000e-04f,
            -7.014409000e-03f,  -3.895883000e-03f,  -7.773570000e-04f,  -7.032202000e-03f,  -3.905766000e-03f,  -7.793300000e-04f,
            -7.049995000e-03f,  -3.915649000e-03f,  -7.813030000e-04f,  -7.067788000e-03f,  -3.925532000e-03f,  -7.832760000e-04f,
            -7.085581000e-03f,  -3.935415000e-03f,  -7.852490000e-04f,  -7.103374000e-03f,  -3.945298000e-03f,  -7.872220000e-04f,
            -7.121167000e-03f,  -3.955181000e-03f,  -7.891950000e-04f,  -7.138960000e-03f,  -3.965064000e-03f,  -7.911680000e-04f,
            -7.156753000e-03f,  -3.974947000e-03f,  -7.931410000e-04f,  -7.174546000e-03f,  -3.984830000e-03f,  -7.951140000e-04f,
            -7.192339000e-03f,  -3.994713000e-03f,  -7.970870000e-04f,  -7.210132000e-03f,  -4.004596000e-03f,  -7.990600000e-04f,
            -7.227925000e-03f,  -4.014479000e-03f,  -8.010330000e-04f,  -7.245718000e-03f,  -4.024362000e-03f,  -8.030060000e-04f,
            -7.263511000e-03f,  -4.034245000e-03f,  -8.049790000e-04f,  -7.281304000e-03f,  -4.044128000e-03f,  -8.069520000e-04f,
            -7.299097000e-03f,  -4.054011000e-03f,  -8.089250000e-04f,  -7.316890000e-03f,  -4.063894000e-03f,  -8.108980000e-04f,
            -7.334683000e-03f,  -4.073777000e-03f,  -8.128710000e-04f,  -7.352476000e-03f,  -4.083660000e-03f,  -8.148440000e-04f,
            -7.370269000e-03f,  -4.093543000e-03f,  -8.168170000e-04f,  -7.388062000e-03f,  -4.103426000e-03f,  -8.187900000e-04f,
            -7.405855000e-03f,  -4.113309000e-03f,  -8.207630000e-04f,  -7.423648000e-03f,  -4.123192000e-03f,  -8.227360000e-04f,
            -7.441441000e-03f,  -4.133075000e-03f,  -8.247090000e-04f,  -7.459234000e-03f,  -4.142958000e-03f,  -8.266820000e-04f,
        };

        float[] gw_expect = new float[] {
            -1.456526627e+02f,  -1.465818371e+02f,  -1.458262648e+02f,  -1.467566524e+02f,  -1.459998670e+02f,  -1.469314678e+02f,
        };
    }
}
