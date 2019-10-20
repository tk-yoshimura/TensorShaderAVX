using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShaderAvxBackend;

namespace TensorShaderAvxBackendTest {
    [TestClass]
    public class UnitTest {
        [TestMethod]
        public void SupportTest() {
            Console.WriteLine($"AVX Support : {Util.IsSupportedAVX}");
            Console.WriteLine($"AVX2 Support : {Util.IsSupportedAVX2}");
            Console.WriteLine($"AVX512 Foundation Support : {Util.IsSupportedAVX512F}");
            Console.WriteLine($"FMA Foundation Support : {Util.IsSupportedFMA}");
        }
    }
}
