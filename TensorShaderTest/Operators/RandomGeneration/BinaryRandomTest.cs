using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShader;
using TensorShader.Operators.RandomGeneration;

namespace TensorShaderTest.Operators.RandomGeneration {
    [TestClass]
    public class BinaryRandomTest {
        [TestMethod]
        public void ExecuteTest() {
            int length = 65535 * 13;

            Shape shape = Shape.Vector(length);

            OverflowCheckedTensor v1 = new OverflowCheckedTensor(shape);

            for (decimal prob = 0; prob <= 1; prob += 0.05m) {
                BinaryRandom ope = new BinaryRandom(shape, new Random(1234), (float)prob);

                ope.Execute(v1);

                float[] y = v1.State;

                Assert.AreEqual((float)prob, y.Average(), 1e-3f);
            }
        }

        [TestMethod]
        public void SpeedTest() {
            int length = 65535 * 16;

            Shape shape = Shape.Vector(length);

            OverflowCheckedTensor v1 = new OverflowCheckedTensor(shape);

            BinaryRandom ope = new BinaryRandom(shape, new Random(1234), 0.25f);

            Stopwatch sw = new Stopwatch();

            sw.Start();

            ope.Execute(v1);
            ope.Execute(v1);
            ope.Execute(v1);
            ope.Execute(v1);

            sw.Stop();

            Console.WriteLine($"{sw.ElapsedMilliseconds / 4} msec");
        }
    }
}
