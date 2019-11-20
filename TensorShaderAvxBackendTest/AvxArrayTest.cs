using System;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using TensorShaderAvxBackend;

namespace TensorShaderAvxBackendTest {
    [TestClass]
    public class AvxArrayTest {
        [TestMethod]
        public void CreateTest() {
            const int length = 15;

            AvxArray<float> arr = new AvxArray<float>(length);

            Assert.IsTrue(arr.IsValid);
            Assert.AreEqual(Marshal.SizeOf(typeof(float)) * length, (int)arr.ByteSize);

            arr.Dispose();

            Assert.IsFalse(arr.IsValid);
            Assert.AreEqual(0, (int)arr.ByteSize);
        }

        [TestMethod]
        public void InitializeTest() {
            const int length = 15;

            float[] v = new float[length];

            AvxArray<float> arr = new AvxArray<float>(length);

            arr.Read(v);

            foreach(float f in v) { 
                Assert.AreEqual(0.0f, f);
            }
        }

        [TestMethod]
        public void WriteReadTest() {
            const int length = 15;

            float[] v = (new float[length]).Select((_, idx) => (float)idx).ToArray();
            float[] v2 = new float[length];

            AvxArray<float> arr = new AvxArray<float>(length);
            AvxArray<float> arr2 = new AvxArray<float>(length);

            arr.Write(v);

            AvxArray<float>.Copy(arr, arr2, length);

            arr2.Read(v2);

            CollectionAssert.AreEqual(v, v2);
        }

        [TestMethod]
        public void RegionWriteReadTest() {
            const int length = 15;

            float[] v = (new float[length]).Select((_, idx) => (float)idx).ToArray();
            float[] v2 = new float[length];
            float[] v3 = (new float[length]).Select((_, idx) => idx < (float)(length - 1) ? (float)idx : 0).ToArray();

            AvxArray<float> arr = new AvxArray<float>(length);
            AvxArray<float> arr2 = new AvxArray<float>(length);

            arr.Write(v, length - 1);

            AvxArray<float>.Copy(arr, arr2, length);

            arr2.Read(v2, length - 1);

            CollectionAssert.AreEqual(v2, v3);
        }

        [TestMethod]
        public void ZerosetTest() {
            const int length = 15;

            float[] v = (new float[length]).Select((_, idx) => (float)idx).ToArray();
            float[] v2 = new float[length];
            float[] v3 = (new float[length]).Select((_, idx) => idx < (float)(length - 1) ? 0 : (float)idx).ToArray();
            float[] v4 = (new float[length]).Select((_, idx) => idx >= 3 && idx < (float)(length - 1) ? 0 : (float)idx).ToArray();

            float[] v5 = new float[length];

            AvxArray<float> arr = new AvxArray<float>(length);
            
            arr.Write(v, length);
            arr.Zeroset();
            arr.Read(v5, length);
            CollectionAssert.AreEqual(v2, v5);

            arr.Write(v, length);
            arr.Zeroset(length - 1);
            arr.Read(v5, length);
            CollectionAssert.AreEqual(v3, v5);

            arr.Write(v, length);
            arr.Zeroset(3, length - 4);
            arr.Read(v5, length);
            CollectionAssert.AreEqual(v4, v5);
        }

        [TestMethod]
        public void CopyTest() {
            const int length = 15;

            float[] v = (new float[length]).Select((_, idx) => (float)idx).ToArray();
            float[] v3 = (new float[length]).Select((_, idx) => idx < (float)(length - 1) ? (float)idx : 0).ToArray();
            float[] v4 = (new float[length]).Select((_, idx) => idx >= 3 && idx < (float)(length - 1) ? (float)idx : 0).ToArray();
            float[] v5 = (new float[length]).Select((_, idx) => idx >= 2 && idx < (float)(length - 2) ? (float)idx + 1 : 0).ToArray();

            float[] v6 = new float[length];

            AvxArray<float> arr = new AvxArray<float>(v);
            AvxArray<float> arr2 = new AvxArray<float>(length);
            
            arr2.Zeroset();
            arr.CopyTo(arr2, arr.Length);
            arr2.Read(v6);
            CollectionAssert.AreEqual(v, v6);

            arr2.Zeroset();
            arr.CopyTo(arr2, arr.Length - 1);
            arr2.Read(v6);
            CollectionAssert.AreEqual(v3, v6);

            arr2.Zeroset();
            arr.CopyTo(3, arr2, 3, arr.Length - 4);
            arr2.Read(v6);
            CollectionAssert.AreEqual(v4, v6);

            arr2.Zeroset();
            arr.CopyTo(3, arr2, 2, arr.Length - 4);
            arr2.Read(v6);
            CollectionAssert.AreEqual(v5, v6);
        }

        [TestMethod]
        public void IndexerTest() {
            AvxArray<float> arr = new float[]{ 1, 2, 3, 4, 5, 6, 7, 8 };

            arr[5] = 9;

            Assert.AreEqual(1, arr[0]);
            Assert.AreEqual(2, arr[1]);
            Assert.AreEqual(3, arr[2]);
            Assert.AreEqual(4, arr[3]);
            Assert.AreEqual(5, arr[4]);
            Assert.AreEqual(9, arr[5]);
            Assert.AreEqual(7, arr[6]);
            Assert.AreEqual(8, arr[7]);

            Assert.ThrowsException<IndexOutOfRangeException>(() => {
                arr[8] = 0;
            });

            Assert.ThrowsException<IndexOutOfRangeException>(() => {
                float v = arr[8];
            });
        }
    }
}
