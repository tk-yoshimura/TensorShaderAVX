using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using static TensorShader.Field;

namespace TensorShaderTest.Links.ArrayManipulation {
    [TestClass]
    public class ExtractChannelTest {
        [TestMethod]
        public void ReferenceTest() {
            Shape inshape = new Shape(ShapeType.Map, 85, 32);
            Shape outshape = new Shape(ShapeType.Map, 23, 32);

            float[] xval = (new float[inshape.Length]).Select((_, idx) => (float)(idx)).Reverse().ToArray();

            float[] tval = (new float[outshape.Length]).Select((_, idx) => (float)(idx * 2)).ToArray();

            Tensor xtensor = new Tensor(inshape, xval);
            Tensor ttensor = new Tensor(outshape, tval);

            ParameterField x = xtensor;
            VariableField t = ttensor;

            Field y = ExtractChannel(x, 13, 23);

            StoreField o = y.Save();

            Field err = y - t;

            (Flow flow, Parameters Parameters) = Flow.Optimize(err);

            flow.Execute();

            float[] gx_actual = x.GradTensor.State;

            AssertError.Tolerance(gx_expect, gx_actual, 1e-7f, 1e-5f, $"not equal gx");
        }

        float[] gx_expect = {
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2706,  2703,  2700,
            2697,  2694,  2691,  2688,  2685,  2682,  2679,  2676,  2673,  2670,  2667,  2664,  2661,  2658,  2655,  2652,
            2649,  2646,  2643,  2640,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  2575,  2572,  2569,  2566,  2563,  2560,  2557,  2554,  2551,  2548,  2545,  2542,  2539,  2536,
            2533,  2530,  2527,  2524,  2521,  2518,  2515,  2512,  2509,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  2444,  2441,  2438,  2435,  2432,  2429,  2426,  2423,  2420,
            2417,  2414,  2411,  2408,  2405,  2402,  2399,  2396,  2393,  2390,  2387,  2384,  2381,  2378,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2313,  2310,  2307,  2304,
            2301,  2298,  2295,  2292,  2289,  2286,  2283,  2280,  2277,  2274,  2271,  2268,  2265,  2262,  2259,  2256,
            2253,  2250,  2247,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  2182,  2179,  2176,  2173,  2170,  2167,  2164,  2161,  2158,  2155,  2152,  2149,  2146,  2143,  2140,
            2137,  2134,  2131,  2128,  2125,  2122,  2119,  2116,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  2051,  2048,  2045,  2042,  2039,  2036,  2033,  2030,  2027,  2024,
            2021,  2018,  2015,  2012,  2009,  2006,  2003,  2000,  1997,  1994,  1991,  1988,  1985,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1920,  1917,  1914,  1911,  1908,
            1905,  1902,  1899,  1896,  1893,  1890,  1887,  1884,  1881,  1878,  1875,  1872,  1869,  1866,  1863,  1860,
            1857,  1854,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            1789,  1786,  1783,  1780,  1777,  1774,  1771,  1768,  1765,  1762,  1759,  1756,  1753,  1750,  1747,  1744,
            1741,  1738,  1735,  1732,  1729,  1726,  1723,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  1658,  1655,  1652,  1649,  1646,  1643,  1640,  1637,  1634,  1631,  1628,
            1625,  1622,  1619,  1616,  1613,  1610,  1607,  1604,  1601,  1598,  1595,  1592,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1527,  1524,  1521,  1518,  1515,  1512,
            1509,  1506,  1503,  1500,  1497,  1494,  1491,  1488,  1485,  1482,  1479,  1476,  1473,  1470,  1467,  1464,
            1461,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1396,
            1393,  1390,  1387,  1384,  1381,  1378,  1375,  1372,  1369,  1366,  1363,  1360,  1357,  1354,  1351,  1348,
            1345,  1342,  1339,  1336,  1333,  1330,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  1265,  1262,  1259,  1256,  1253,  1250,  1247,  1244,  1241,  1238,  1235,  1232,
            1229,  1226,  1223,  1220,  1217,  1214,  1211,  1208,  1205,  1202,  1199,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  1134,  1131,  1128,  1125,  1122,  1119,  1116,
            1113,  1110,  1107,  1104,  1101,  1098,  1095,  1092,  1089,  1086,  1083,  1080,  1077,  1074,  1071,  1068,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1003,  1000,
            997,  994,  991,  988,  985,  982,  979,  976,  973,  970,  967,  964,  961,  958,  955,  952,
            949,  946,  943,  940,  937,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  872,  869,  866,  863,  860,  857,  854,  851,  848,  845,  842,  839,  836,
            833,  830,  827,  824,  821,  818,  815,  812,  809,  806,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  741,  738,  735,  732,  729,  726,  723,  720,
            717,  714,  711,  708,  705,  702,  699,  696,  693,  690,  687,  684,  681,  678,  675,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  610,  607,  604,
            601,  598,  595,  592,  589,  586,  583,  580,  577,  574,  571,  568,  565,  562,  559,  556,
            553,  550,  547,  544,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  479,  476,  473,  470,  467,  464,  461,  458,  455,  452,  449,  446,  443,  440,
            437,  434,  431,  428,  425,  422,  419,  416,  413,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  348,  345,  342,  339,  336,  333,  330,  327,  324,
            321,  318,  315,  312,  309,  306,  303,  300,  297,  294,  291,  288,  285,  282,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  217,  214,  211,  208,
            205,  202,  199,  196,  193,  190,  187,  184,  181,  178,  175,  172,  169,  166,  163,  160,
            157,  154,  151,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  86,  83,  80,  77,  74,  71,  68,  65,  62,  59,  56,  53,  50,  47,  44,
            41,  38,  35,  32,  29,  26,  23,  20,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  -45,  -48,  -51,  -54,  -57,  -60,  -63,  -66,  -69,  -72,
            -75,  -78,  -81,  -84,  -87,  -90,  -93,  -96,  -99,  -102,  -105,  -108,  -111,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  -176,  -179,  -182,  -185,  -188,
            -191,  -194,  -197,  -200,  -203,  -206,  -209,  -212,  -215,  -218,  -221,  -224,  -227,  -230,  -233,  -236,
            -239,  -242,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            -307,  -310,  -313,  -316,  -319,  -322,  -325,  -328,  -331,  -334,  -337,  -340,  -343,  -346,  -349,  -352,
            -355,  -358,  -361,  -364,  -367,  -370,  -373,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  -438,  -441,  -444,  -447,  -450,  -453,  -456,  -459,  -462,  -465,  -468,
            -471,  -474,  -477,  -480,  -483,  -486,  -489,  -492,  -495,  -498,  -501,  -504,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  -569,  -572,  -575,  -578,  -581,  -584,
            -587,  -590,  -593,  -596,  -599,  -602,  -605,  -608,  -611,  -614,  -617,  -620,  -623,  -626,  -629,  -632,
            -635,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  -700,
            -703,  -706,  -709,  -712,  -715,  -718,  -721,  -724,  -727,  -730,  -733,  -736,  -739,  -742,  -745,  -748,
            -751,  -754,  -757,  -760,  -763,  -766,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  -831,  -834,  -837,  -840,  -843,  -846,  -849,  -852,  -855,  -858,  -861,  -864,
            -867,  -870,  -873,  -876,  -879,  -882,  -885,  -888,  -891,  -894,  -897,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  -962,  -965,  -968,  -971,  -974,  -977,  -980,
            -983,  -986,  -989,  -992,  -995,  -998,  -1001,  -1004,  -1007,  -1010,  -1013,  -1016,  -1019,  -1022,  -1025,  -1028,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  -1093,  -1096,
            -1099,  -1102,  -1105,  -1108,  -1111,  -1114,  -1117,  -1120,  -1123,  -1126,  -1129,  -1132,  -1135,  -1138,  -1141,  -1144,
            -1147,  -1150,  -1153,  -1156,  -1159,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  -1224,  -1227,  -1230,  -1233,  -1236,  -1239,  -1242,  -1245,  -1248,  -1251,  -1254,  -1257,  -1260,
            -1263,  -1266,  -1269,  -1272,  -1275,  -1278,  -1281,  -1284,  -1287,  -1290,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  -1355,  -1358,  -1361,  -1364,  -1367,  -1370,  -1373,  -1376,
            -1379,  -1382,  -1385,  -1388,  -1391,  -1394,  -1397,  -1400,  -1403,  -1406,  -1409,  -1412,  -1415,  -1418,  -1421,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        };
    }
}
