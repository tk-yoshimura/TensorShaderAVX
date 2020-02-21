#pragma once

#pragma warning(disable: 4793)

#include <immintrin.h>

using namespace System;
using namespace System::Runtime::CompilerServices;

namespace TensorShaderAvxBackend {
    [assembly:InternalsVisibleTo("TensorShaderAvxBackendTest")] ;

    extern __m256i masktable_m256(int k);
    extern __m128i masktable_m128(int k);

    generic <typename T> where T : ValueType
    public ref class AvxArray {
        private:
        IntPtr ptr;
        UInt64 length;

        internal:
        property IntPtr Ptr { IntPtr get(); }

        public:
        property UInt64 Length { UInt64 get(); }
        static property UInt64 MaxLength { UInt64 get(); }
        property UInt64 ByteSize { UInt64 get(); }
        static property UInt64 MaxByteSize { UInt64 get(); }
        static property UInt64 ElementSize { UInt64 get(); }

        static property UInt64 Alignment { UInt64 get(); }

        property bool IsValid { bool get(); }
        static property Type^ ElementType { Type^ get(); }

        property cli::array<T>^ Value { cli::array<T>^ get(); }

        property String^ Overview { String^ get(); }

        AvxArray(UInt64 length);
        AvxArray(cli::array<T>^ array);

        property T default[UInt64] { T get(UInt64 index); void set(UInt64 index, T value); }

        static operator AvxArray^(cli::array<T>^ array);
        static operator cli::array<T>^(AvxArray^ array);

        void Write(cli::array<T>^ array);
        void Read(cli::array<T>^ array);

        void Write(cli::array<T>^ array, UInt64 count);
        void Read(cli::array<T>^ array, UInt64 count);

        void Zeroset();
        void Zeroset(UInt64 count);
        void Zeroset(UInt64 index, UInt64 count);

        void CopyTo(AvxArray^ array, UInt64 count);
        void CopyTo(UInt64 index, AvxArray^ dst_array, UInt64 dst_index, UInt64 count);

        static void Copy(AvxArray^ src_array, AvxArray^ dst_array, UInt64 count);
        static void Copy(AvxArray^ src_array, UInt64 src_index, AvxArray^ dst_array, UInt64 dst_index, UInt64 count);

        String^ ToString() override;

        ~AvxArray();

        !AvxArray();
    };

    public ref class Util abstract sealed {
        internal:
        static void CheckLength(unsigned int length, ... cli::array<AvxArray<float>^>^ arrays);
        static void CheckOutOfRange(unsigned int index, unsigned int length, ... cli::array<AvxArray<float>^>^ arrays);
        static void CheckDuplicateArray(... cli::array<AvxArray<float>^>^ arrays);

        public:
        static property bool IsSupportedAVX { bool get(); };
        static property bool IsSupportedAVX2 { bool get(); };
        static property bool IsSupportedAVX512F { bool get(); };
        static property bool IsSupportedFMA { bool get(); };
    };

    public ref class Elementwise abstract sealed {
        static Elementwise();

        public:
        static void Add(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);
        static void Sub(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);
        static void Mul(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);
        static void Div(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);
        static void Pow(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);
        static void SignedPow(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);

        static void ArcTan2(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);

        static void AddConstant(unsigned int length, float c, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void SubConstant(unsigned int length, float c, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void MulConstant(unsigned int length, float c, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void DivConstant(unsigned int length, float c, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void PowConstant(unsigned int length, float c, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void SignedPowConstant(unsigned int length, float c, AvxArray<float>^ src, AvxArray<float>^ dst);

        static void Maximum(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);
        static void Minimum(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);
        static void Clamp(unsigned int length, AvxArray<float>^ srcval, AvxArray<float>^ srcmin, AvxArray<float>^ srcmax, AvxArray<float>^ dst);

        static void MaximumConstant(unsigned int length, float c, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void MinimumConstant(unsigned int length, float c, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void ClampConstant(unsigned int length, float cmin, float cmax, AvxArray<float>^ src, AvxArray<float>^ dst);
        
        static void GreaterThan(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);
        static void GreaterThanOrEqual(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);

        static void LessThan(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);
        static void LessThanOrEqual(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);

        static void Equal(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);
        static void NotEqual(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);

        static void GreaterThanConstant(unsigned int length, float c, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void GreaterThanOrEqualConstant(unsigned int length, float c, AvxArray<float>^ src, AvxArray<float>^ dst);

        static void LessThanConstant(unsigned int length, float c, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void LessThanOrEqualConstant(unsigned int length, float c, AvxArray<float>^ src, AvxArray<float>^ dst);

        static void EqualConstant(unsigned int length, float c, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void NotEqualConstant(unsigned int length, float c, AvxArray<float>^ src, AvxArray<float>^ dst);

        static void LogicalNot(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void LogicalAnd(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);
        static void LogicalOr(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);
        static void LogicalXor(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);

        static void Abs(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void Sign(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void Step(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        
        static void Neg(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void Rcp(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);

        static void Square(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void Cube(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);

        static void Sqrt(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void Rsqrt(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void SignedSqrt(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);

        static void Cbrt(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void Rcbrt(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        
        static void Sin(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void Cos(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void Tan(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);

        static void Sinh(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void Cosh(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void Tanh(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);

        static void ArcSin(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void ArcCos(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void ArcTan(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);

        static void Exp(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void Exp2(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        
        static void Log(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void Log2(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void Log10(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);

        static void Floor(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void Ceil(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void Round(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);

        static void Sigmoid(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void SoftPlus(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void LogCosh(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);

        static void IsNan(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void NanAsZero(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);

        static void Relu(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void ReluGrad(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);

        static void LeakyRelu(unsigned int length, float slope, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void LeakyReluGrad(unsigned int length, float slope, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);

        static void Elu(unsigned int length, float slope, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void EluGrad(unsigned int length, float slope, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);

        static void Sum(unsigned int length, cli::array <AvxArray<float>^> ^src, AvxArray<float>^ dst);

        static void Lerp(unsigned int length, AvxArray<float>^ srccondition, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);
    };

    public ref class Channelwise abstract sealed {
        static Channelwise();

        public:
        static void Add(unsigned int vector_length, unsigned int map_length, AvxArray<float>^ srcvector, AvxArray<float>^ srcmap, AvxArray<float>^ dstmap);
        static void Mul(unsigned int vector_length, unsigned int map_length, AvxArray<float>^ srcvector, AvxArray<float>^ srcmap, AvxArray<float>^ dstmap);

        static void SubLVector(unsigned int vector_length, unsigned int map_length, AvxArray<float>^ srcvector, AvxArray<float>^ srcmap, AvxArray<float>^ dstmap);
        static void SubRVector(unsigned int vector_length, unsigned int map_length, AvxArray<float>^ srcvector, AvxArray<float>^ srcmap, AvxArray<float>^ dstmap);

        static void DivLVector(unsigned int vector_length, unsigned int map_length, AvxArray<float>^ srcvector, AvxArray<float>^ srcmap, AvxArray<float>^ dstmap);
        static void DivRVector(unsigned int vector_length, unsigned int map_length, AvxArray<float>^ srcvector, AvxArray<float>^ srcmap, AvxArray<float>^ dstmap);
    };

    public ref class Batchwise abstract sealed {
        static Batchwise();

        public:
        static void Mul(unsigned int vector_length, unsigned int map_length, AvxArray<float>^ srcvector, AvxArray<float>^ srcmap, AvxArray<float>^ dstmap);
    };

    public ref class ArrayManipulation abstract sealed {
        static ArrayManipulation();
        
        public:
        static void Clear(unsigned int length, float c, AvxArray<float>^ dst);
        
        static void PatternCopy(unsigned int src_stride, unsigned int src_index, unsigned int dst_stride, unsigned int dst_index, unsigned int copy_length, unsigned int slides, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void PatternCopy(unsigned int src_offset, unsigned int src_stride, unsigned int src_index, unsigned int dst_offset, unsigned int dst_stride, unsigned int dst_index, unsigned int copy_length, unsigned int slides, AvxArray<float>^ src, AvxArray<float>^ dst);
                
        static void Broadcast(unsigned int src_length, AvxArray<float>^ src, unsigned int dst_length, AvxArray<float>^ dst, unsigned int slides);
        
        static void Flip(unsigned int stride, unsigned int axislength, unsigned int slides, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void Sort(unsigned int stride, unsigned int axislength, unsigned int slides, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void SortWithKey(unsigned int stride, unsigned int axislength, unsigned int slides, AvxArray<float>^ src_key, AvxArray<float>^ src_value, AvxArray<float>^ dst_key, AvxArray<float>^ dst_value);
    };

    public ref class Indexer abstract sealed {
        static Indexer();

        public:
        static void OneHotVector(unsigned int length, unsigned int channels, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void ArgMin(unsigned int length, unsigned int channels, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void ArgMax(unsigned int length, unsigned int channels, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void Index(unsigned int stride, unsigned int axislength, unsigned int clones, AvxArray<float>^ dst);
    };

    public ref class Randomize abstract sealed {
        public:
        static void Uniform(unsigned int length, AvxArray<float>^ dst, Random ^random);
        static void Normal(unsigned int length, AvxArray<float>^ dst, Random^ random);
        static void Bernoulli(unsigned int length, double prob, AvxArray<float>^ dst, Random^ random);
    };

    public ref class Aggregation abstract sealed {
        static Aggregation();

        public:
        static void Sum(unsigned int src_length, AvxArray<float>^ src, unsigned int dst_length, AvxArray<float>^ dst, unsigned int slides);
        static void Average(unsigned int src_length, AvxArray<float>^ src, unsigned int dst_length, AvxArray<float>^ dst, unsigned int slides);
        static void Min(unsigned int src_length, AvxArray<float>^ src, unsigned int dst_length, AvxArray<float>^ dst, unsigned int slides);
        static void Max(unsigned int src_length, AvxArray<float>^ src, unsigned int dst_length, AvxArray<float>^ dst, unsigned int slides);
    };

    public ref class Pool abstract sealed {
        static Pool();

        public:
        static void MaxPool1D(unsigned int channels, unsigned int inwidth, unsigned int batch, unsigned int stride, AvxArray<float>^ inmap, AvxArray<float>^ outmap);
        static void MaxPool2D(unsigned int channels, unsigned int inwidth, unsigned int inheight, unsigned int batch, unsigned int stride, AvxArray<float>^ inmap, AvxArray<float>^ outmap);
        static void MaxPool3D(unsigned int channels, unsigned int inwidth, unsigned int inheight, unsigned int indepth, unsigned int batch, unsigned int stride, AvxArray<float>^ inmap, AvxArray<float>^ outmap);

        static void AveragePool1D(unsigned int channels, unsigned int inwidth, unsigned int batch, unsigned int stride, AvxArray<float>^ inmap, AvxArray<float>^ outmap);
        static void AveragePool2D(unsigned int channels, unsigned int inwidth, unsigned int inheight, unsigned int batch, unsigned int stride, AvxArray<float>^ inmap, AvxArray<float>^ outmap);
        static void AveragePool3D(unsigned int channels, unsigned int inwidth, unsigned int inheight, unsigned int indepth, unsigned int batch, unsigned int stride, AvxArray<float>^ inmap, AvxArray<float>^ outmap);

        static void MaxUnpool1D(unsigned int channels, unsigned int outwidth, unsigned int batch, unsigned int stride, AvxArray<float>^ ingrad, AvxArray<float>^ inpool, AvxArray<float>^ inmap, AvxArray<float>^ outmap);
        static void MaxUnpool2D(unsigned int channels, unsigned int outwidth, unsigned int outheight, unsigned int batch, unsigned int stride, AvxArray<float>^ ingrad, AvxArray<float>^ inpool, AvxArray<float>^ inmap, AvxArray<float>^ outmap);
        static void MaxUnpool3D(unsigned int channels, unsigned int outwidth, unsigned int outheight, unsigned int outdepth, unsigned int batch, unsigned int stride, AvxArray<float>^ ingrad, AvxArray<float>^ inpool, AvxArray<float>^ inmap, AvxArray<float>^ outmap);

        static void AverageUnpool1D(unsigned int channels, unsigned int outwidth, unsigned int batch, unsigned int stride, AvxArray<float>^ inmap, AvxArray<float>^ outmap);
        static void AverageUnpool2D(unsigned int channels, unsigned int outwidth, unsigned int outheight, unsigned int batch, unsigned int stride, AvxArray<float>^ inmap, AvxArray<float>^ outmap);
        static void AverageUnpool3D(unsigned int channels, unsigned int outwidth, unsigned int outheight, unsigned int outdepth, unsigned int batch, unsigned int stride, AvxArray<float>^ inmap, AvxArray<float>^ outmap);
    };

    public ref class Transform abstract sealed {
        static Transform();

        public:
        static void ChannelToSpace2D(unsigned int outchannels, unsigned int inwidth, unsigned int inheight, unsigned int batch, unsigned int scale, AvxArray<float>^ inmap, AvxArray<float>^ outmap);
        static void SpaceToChannel2D(unsigned int inchannels, unsigned int outwidth, unsigned int outheight, unsigned int batch, unsigned int scale, AvxArray<float>^ inmap, AvxArray<float>^ outmap);

        static void ChannelToSpace3D(unsigned int outchannels, unsigned int inwidth, unsigned int inheight, unsigned int indepth, unsigned int batch, unsigned int scale, AvxArray<float>^ inmap, AvxArray<float>^ outmap);
        static void SpaceToChannel3D(unsigned int inchannels, unsigned int outwidth, unsigned int outheight, unsigned int outdepth, unsigned int batch, unsigned int scale, AvxArray<float>^ inmap, AvxArray<float>^ outmap);
    };

    public ref class Padding abstract sealed {
        static Padding();

        public:
        static void ZeroPadding1D(unsigned int channels, unsigned int inwidth, 
                                  unsigned int batch, 
                                  unsigned int pad_left, unsigned int pad_right, 
                                  AvxArray<float>^ inmap, AvxArray<float>^ outmap);

        static void EdgePadding1D(unsigned int channels, unsigned int inwidth, 
                                  unsigned int batch, 
                                  unsigned int pad_left, unsigned int pad_right, 
                                  AvxArray<float>^ inmap, AvxArray<float>^ outmap);

        static void ZeroPadding2D(unsigned int channels, unsigned int inwidth, unsigned int inheight, 
                                  unsigned int batch, 
                                  unsigned int pad_left, unsigned int pad_right, 
                                  unsigned int pad_top, unsigned int pad_bottom,
                                  AvxArray<float>^ inmap, AvxArray<float>^ outmap);

        static void EdgePadding2D(unsigned int channels, unsigned int inwidth, unsigned int inheight,
                                  unsigned int batch, 
                                  unsigned int pad_left, unsigned int pad_right,
                                  unsigned int pad_top, unsigned int pad_bottom,
                                  AvxArray<float>^ inmap, AvxArray<float>^ outmap);

        static void ZeroPadding3D(unsigned int channels, unsigned int inwidth, unsigned int inheight, unsigned int indepth,
                                  unsigned int batch, 
                                  unsigned int pad_left, unsigned int pad_right,
                                  unsigned int pad_top, unsigned int pad_bottom,
                                  unsigned int pad_front, unsigned int pad_rear,
                                  AvxArray<float>^ inmap, AvxArray<float>^ outmap);

        static void EdgePadding3D(unsigned int channels, unsigned int inwidth, unsigned int inheight, unsigned int indepth,
                                  unsigned int batch, 
                                  unsigned int pad_left, unsigned int pad_right,
                                  unsigned int pad_top, unsigned int pad_bottom,
                                  unsigned int pad_front, unsigned int pad_rear,
                                  AvxArray<float>^ inmap, AvxArray<float>^ outmap);
    };

    public ref class Trimming abstract sealed {
        static Trimming();

        public:
        static void Trimming1D(unsigned int channels, unsigned int outwidth,
                               unsigned int batch, 
                               unsigned int trim_left, unsigned int trim_right,
                               AvxArray<float>^ inmap, AvxArray<float>^ outmap);

        static void Trimming2D(unsigned int channels, unsigned int outwidth, unsigned int outheight,
                               unsigned int batch, 
                               unsigned int trim_left, unsigned int trim_right,
                               unsigned int trim_top, unsigned int trim_bottom,
                               AvxArray<float>^ inmap, AvxArray<float>^ outmap);

        static void Trimming3D(unsigned int channels, unsigned int outwidth, unsigned int outheight, unsigned int outdepth,
                               unsigned int batch, 
                               unsigned int trim_left, unsigned int trim_right,
                               unsigned int trim_top, unsigned int trim_bottom,
                               unsigned int trim_front, unsigned int trim_rear,
                               AvxArray<float>^ inmap, AvxArray<float>^ outmap);
    };

    public ref class Zoom abstract sealed {
        static Zoom();

        public:
        static void NeighborZoom1D(unsigned int channels, unsigned int inwidth,
                                   unsigned int batch, 
                                   AvxArray<float>^ inmap, AvxArray<float>^ outmap);

        static void NeighborZoom2D(unsigned int channels, unsigned int inwidth, unsigned int inheight,
                                   unsigned int batch, 
                                   AvxArray<float>^ inmap, AvxArray<float>^ outmap);

        static void NeighborZoom3D(unsigned int channels, unsigned int inwidth, unsigned int inheight, unsigned int indepth,
                                   unsigned int batch, 
                                   AvxArray<float>^ inmap, AvxArray<float>^ outmap);

        static void LinearZoom1D(unsigned int channels, unsigned int inwidth,
                                 unsigned int batch, 
                                 AvxArray<float>^ inmap, AvxArray<float>^ outmap);

        static void LinearZoom2D(unsigned int channels, unsigned int inwidth, unsigned int inheight,
                                 unsigned int batch, 
                                 AvxArray<float>^ inmap, AvxArray<float>^ outmap);

        static void LinearZoom3D(unsigned int channels, unsigned int inwidth, unsigned int inheight, unsigned int indepth,
                                 unsigned int batch, 
                                 AvxArray<float>^ inmap, AvxArray<float>^ outmap);
    };

    public ref class Convolution abstract sealed {
        static Convolution();

        public:
        static void Dense(unsigned int inchannels, unsigned int outchannels, 
                          unsigned int batch, 
                          AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void TransposeDense(unsigned int inchannels, unsigned int outchannels, 
                                   unsigned int batch, 
                                   AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void KernelProductDense(unsigned int inchannels, unsigned int outchannels, 
                                       unsigned int batch, 
                                       AvxArray<float>^ inmap, AvxArray<float>^ outmap, AvxArray<float>^ kernel);

        static void Convolution1D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth,
                                  unsigned int batch, unsigned int kwidth, 
                                  AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void Deconvolution1D(unsigned int inchannels, unsigned int outchannels, unsigned int outwidth,
                                    unsigned int batch, unsigned int kwidth, 
                                    AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);
        
        static void KernelProduct1D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth,
                                    unsigned int batch, unsigned int kwidth, 
                                    AvxArray<float>^ inmap, AvxArray<float>^ outmap, AvxArray<float>^ kernel);

        static void Convolution2D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth, unsigned int inheight,
                                  unsigned int batch, unsigned int kwidth, unsigned int kheight, 
                                  AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void Deconvolution2D(unsigned int inchannels, unsigned int outchannels, unsigned int outwidth, unsigned int outheight,
                                    unsigned int batch, unsigned int kwidth, unsigned int kheight, 
                                    AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void KernelProduct2D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth, unsigned int inheight,
                                    unsigned int batch, unsigned int kwidth, unsigned int kheight, 
                                    AvxArray<float>^ inmap, AvxArray<float>^ outmap, AvxArray<float>^ kernel);

        static void Convolution3D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth, unsigned int inheight, unsigned int indepth,
                                  unsigned int batch, unsigned int kwidth, unsigned int kheight, unsigned int kdepth, 
                                  AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void Deconvolution3D(unsigned int inchannels, unsigned int outchannels, unsigned int outwidth, unsigned int outheight, unsigned int outdepth,
                                    unsigned int batch, unsigned int kwidth, unsigned int kheight, unsigned int kdepth, 
                                    AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void KernelProduct3D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth, unsigned int inheight, unsigned int indepth,
                                    unsigned int batch, unsigned int kwidth, unsigned int kheight, unsigned int kdepth, 
                                    AvxArray<float>^ inmap, AvxArray<float>^ outmap, AvxArray<float>^ kernel);

        static void ChannelwiseConvolution1D(unsigned int channels, unsigned int inwidth,
                                             unsigned int batch, unsigned int kwidth, 
                                             AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void ChannelwiseDeconvolution1D(unsigned int channels, unsigned int outwidth,
                                               unsigned int batch, unsigned int kwidth, 
                                               AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void ChannelwiseKernelProduct1D(unsigned int channels, unsigned int inwidth,
                                               unsigned int batch, unsigned int kwidth, 
                                               AvxArray<float>^ inmap, AvxArray<float>^ outmap, AvxArray<float>^ kernel);

        static void ChannelwiseConvolution2D(unsigned int channels, unsigned int inwidth, unsigned int inheight,
                                             unsigned int batch, unsigned int kwidth, unsigned int kheight, 
                                             AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void ChannelwiseDeconvolution2D(unsigned int channels, unsigned int outwidth, unsigned int outheight,
                                               unsigned int batch, unsigned int kwidth, unsigned int kheight,
                                               AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void ChannelwiseKernelProduct2D(unsigned int channels, unsigned int inwidth, unsigned int inheight,
                                               unsigned int batch, unsigned int kwidth, unsigned int kheight, 
                                               AvxArray<float>^ inmap, AvxArray<float>^ outmap, AvxArray<float>^ kernel);

        static void ChannelwiseConvolution3D(unsigned int channels, unsigned int inwidth, unsigned int inheight, unsigned int indepth,
                                             unsigned int batch, unsigned int kwidth, unsigned int kheight, unsigned int kdepth, 
                                             AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void ChannelwiseDeconvolution3D(unsigned int channels, unsigned int outwidth, unsigned int outheight, unsigned int outdepth,
                                               unsigned int batch, unsigned int kwidth, unsigned int kheight, unsigned int kdepth,
                                               AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void ChannelwiseKernelProduct3D(unsigned int channels, unsigned int inwidth, unsigned int inheight, unsigned int indepth,
                                               unsigned int batch, unsigned int kwidth, unsigned int kheight, unsigned int kdepth,
                                               AvxArray<float>^ inmap, AvxArray<float>^ outmap, AvxArray<float>^ kernel);

        static void PointwiseConvolution(unsigned int inchannels, unsigned int outchannels, unsigned int points,
                                         AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void PointwiseDeconvolution(unsigned int inchannels, unsigned int outchannels, unsigned int points,
                                           AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void PointwiseKernelProduct(unsigned int inchannels, unsigned int outchannels, unsigned int points,
                                           AvxArray<float>^ inmap, AvxArray<float>^ outmap, AvxArray<float>^ kernel);
    };

    public ref class Complex abstract sealed {
        static Complex();

        public:
        static void Mul(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);
        static void MulGrad(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);

        static void RRelu(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void RReluGrad(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);

        static void ZRelu(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void ZReluGrad(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);

        static void Conjugate(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        
        static void Square(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void SquareGrad(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);

        static void Decay(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void DecayGrad(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);

        static void Squash(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void SquashGrad(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);

        static void Normalize(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void NormalizeGrad(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);

        static void Cast(unsigned int length, AvxArray<float>^ src_real, AvxArray<float>^ src_imag, AvxArray<float>^ dst);
        static void Real(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst_real);
        static void Imag(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst_imag);
        static void PureReal(unsigned int length, AvxArray<float>^ src_real, AvxArray<float>^ dst);
        static void PureImag(unsigned int length, AvxArray<float>^ src_imag, AvxArray<float>^ dst);

        static void Dense(unsigned int inchannels, unsigned int outchannels,
                          unsigned int batch, bool gradmode,
                          AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void TransposeDense(unsigned int inchannels, unsigned int outchannels,
                                   unsigned int batch, bool gradmode,
                                   AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void KernelProductDense(unsigned int inchannels, unsigned int outchannels,
                                       unsigned int batch, bool transpose,
                                       AvxArray<float>^ inmap, AvxArray<float>^ outmap, AvxArray<float>^ kernel);

        static void Convolution1D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth,
                                  unsigned int batch, unsigned int kwidth, bool gradmode,
                                  AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void Deconvolution1D(unsigned int inchannels, unsigned int outchannels, unsigned int outwidth,
                                    unsigned int batch, unsigned int kwidth, bool gradmode,
                                    AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void KernelProduct1D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth,
                                    unsigned int batch, unsigned int kwidth, bool transpose,
                                    AvxArray<float>^ inmap, AvxArray<float>^ outmap, AvxArray<float>^ kernel);

        static void Convolution2D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth, unsigned int inheight,
                                  unsigned int batch, unsigned int kwidth, unsigned int kheight, bool gradmode,
                                  AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void Deconvolution2D(unsigned int inchannels, unsigned int outchannels, unsigned int outwidth, unsigned int outheight,
                                    unsigned int batch, unsigned int kwidth, unsigned int kheight, bool gradmode,
                                    AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void KernelProduct2D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth, unsigned int inheight,
                                    unsigned int batch, unsigned int kwidth, unsigned int kheight, bool transpose,
                                    AvxArray<float>^ inmap, AvxArray<float>^ outmap, AvxArray<float>^ kernel);

        static void Convolution3D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth, unsigned int inheight, unsigned int indepth,
                                  unsigned int batch, unsigned int kwidth, unsigned int kheight, unsigned int kdepth, bool gradmode,
                                  AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void Deconvolution3D(unsigned int inchannels, unsigned int outchannels, unsigned int outwidth, unsigned int outheight, unsigned int outdepth,
                                    unsigned int batch, unsigned int kwidth, unsigned int kheight, unsigned int kdepth, bool gradmode,
                                    AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void KernelProduct3D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth, unsigned int inheight, unsigned int indepth,
                                    unsigned int batch, unsigned int kwidth, unsigned int kheight, unsigned int kdepth, bool transpose,
                                    AvxArray<float>^ inmap, AvxArray<float>^ outmap, AvxArray<float>^ kernel);
    };

    public ref class Trivector abstract sealed {
        static Trivector();

        public:
        static void Mul(unsigned int vectorlength, AvxArray<float>^ src_vector, AvxArray<float>^ src_quaternion, AvxArray<float>^ dst_vector);
        static void MulQGrad(unsigned int vectorlength, AvxArray<float>^ src_vector_value, AvxArray<float>^ src_vector_grad, AvxArray<float>^ src_quaternion, AvxArray<float>^ dst_quaternion);
        static void MulVGrad(unsigned int vectorlength, AvxArray<float>^ src_vector, AvxArray<float>^ src_quaternion, AvxArray<float>^ dst_vector);

        static void Cross(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);
        
        static void Decay(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void DecayGrad(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);

        static void Squash(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void SquashGrad(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);

        static void Normalize(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void NormalizeGrad(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);

        static void Cast(unsigned int length, AvxArray<float>^ src_x, AvxArray<float>^ src_y, AvxArray<float>^ src_z, AvxArray<float>^ dst);
        static void X(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst_x);
        static void Y(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst_y);
        static void Z(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst_z);
        static void PureX(unsigned int length, AvxArray<float>^ src_x, AvxArray<float>^ dst);
        static void PureY(unsigned int length, AvxArray<float>^ src_y, AvxArray<float>^ dst);
        static void PureZ(unsigned int length, AvxArray<float>^ src_z, AvxArray<float>^ dst);

        static void Dense(unsigned int inchannels, unsigned int outchannels,
                          unsigned int batch, bool gradmode,
                          AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void TransposeDense(unsigned int inchannels, unsigned int outchannels,
                                   unsigned int batch, bool gradmode,
                                   AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void KernelProductDense(unsigned int inchannels, unsigned int outchannels,
                                       unsigned int batch, bool transpose,
                                       AvxArray<float>^ inmap, AvxArray<float>^ outmap, AvxArray<float>^ kernel_value, AvxArray<float>^ kernel_grad);


        static void Convolution1D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth,
                                  unsigned int batch, unsigned int kwidth, bool gradmode,
                                  AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void Deconvolution1D(unsigned int inchannels, unsigned int outchannels, unsigned int outwidth,
                                    unsigned int batch, unsigned int kwidth, bool gradmode,
                                    AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void KernelProduct1D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth,
                                    unsigned int batch, unsigned int kwidth, bool transpose,
                                    AvxArray<float>^ inmap, AvxArray<float>^ outmap, AvxArray<float>^ kernel_value, AvxArray<float>^ kernel_grad);

        static void Convolution2D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth, unsigned int inheight,
                                  unsigned int batch, unsigned int kwidth, unsigned int kheight, bool gradmode,
                                  AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void Deconvolution2D(unsigned int inchannels, unsigned int outchannels, unsigned int outwidth, unsigned int outheight,
                                    unsigned int batch, unsigned int kwidth, unsigned int kheight, bool gradmode,
                                    AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void KernelProduct2D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth, unsigned int inheight,
                                    unsigned int batch, unsigned int kwidth, unsigned int kheight, bool transpose,
                                    AvxArray<float>^ inmap, AvxArray<float>^ outmap, AvxArray<float>^ kernel_value, AvxArray<float>^ kernel_grad);

        static void Convolution3D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth, unsigned int inheight, unsigned int indepth,
                                  unsigned int batch, unsigned int kwidth, unsigned int kheight, unsigned int kdepth, bool gradmode,
                                  AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void Deconvolution3D(unsigned int inchannels, unsigned int outchannels, unsigned int outwidth, unsigned int outheight, unsigned int outdepth,
                                    unsigned int batch, unsigned int kwidth, unsigned int kheight, unsigned int kdepth, bool gradmode,
                                    AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void KernelProduct3D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth, unsigned int inheight, unsigned int indepth,
                                    unsigned int batch, unsigned int kwidth, unsigned int kheight, unsigned int kdepth, bool transpose,
                                    AvxArray<float>^ inmap, AvxArray<float>^ outmap, AvxArray<float>^ kernel_value, AvxArray<float>^ kernel_grad);
    };

    public ref class Quaternion abstract sealed {
        static Quaternion();

        public:
        static void Mul(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);
        static void MulGrad(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);
        static void MulTransposeGrad(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);

        static void RRelu(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void RReluGrad(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);

        static void Conjugate(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);

        static void Square(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        
        static void Decay(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void DecayGrad(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);

        static void Squash(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void SquashGrad(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);

        static void Normalize(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst);
        static void NormalizeGrad(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst);

        static void Cast(unsigned int length, AvxArray<float>^ src_r, AvxArray<float>^ src_i, AvxArray<float>^ src_j, AvxArray<float>^ src_k, AvxArray<float>^ dst);
        static void R(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst_r);
        static void I(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst_i);
        static void J(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst_j);
        static void K(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst_k);
        static void PureR(unsigned int length, AvxArray<float>^ src_r, AvxArray<float>^ dst);
        static void PureI(unsigned int length, AvxArray<float>^ src_i, AvxArray<float>^ dst);
        static void PureJ(unsigned int length, AvxArray<float>^ src_j, AvxArray<float>^ dst);
        static void PureK(unsigned int length, AvxArray<float>^ src_k, AvxArray<float>^ dst);

        static void Dense(unsigned int inchannels, unsigned int outchannels,
                          unsigned int batch, bool gradmode,
                          AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void TransposeDense(unsigned int inchannels, unsigned int outchannels,
                                   unsigned int batch, bool gradmode,
                                   AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void KernelProductDense(unsigned int inchannels, unsigned int outchannels,
                                       unsigned int batch, bool transpose,
                                       AvxArray<float>^ inmap, AvxArray<float>^ outmap, AvxArray<float>^ kernel);

        static void Convolution1D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth,
                                  unsigned int batch, unsigned int kwidth, bool gradmode,
                                  AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void Deconvolution1D(unsigned int inchannels, unsigned int outchannels, unsigned int outwidth,
                                    unsigned int batch, unsigned int kwidth, bool gradmode,
                                    AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void KernelProduct1D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth,
                                    unsigned int batch, unsigned int kwidth, bool transpose,
                                    AvxArray<float>^ inmap, AvxArray<float>^ outmap, AvxArray<float>^ kernel);

        static void Convolution2D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth, unsigned int inheight,
                                  unsigned int batch, unsigned int kwidth, unsigned int kheight, bool gradmode,
                                  AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void Deconvolution2D(unsigned int inchannels, unsigned int outchannels, unsigned int outwidth, unsigned int outheight,
                                    unsigned int batch, unsigned int kwidth, unsigned int kheight, unsigned int stride, bool gradmode,
                                    AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void KernelProduct2D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth, unsigned int inheight,
                                    unsigned int batch, unsigned int kwidth, unsigned int kheight, bool transpose,
                                    AvxArray<float>^ inmap, AvxArray<float>^ outmap, AvxArray<float>^ kernel);

        static void Convolution3D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth, unsigned int inheight, unsigned int indepth,
                                  unsigned int batch, unsigned int kwidth, unsigned int kheight, unsigned int kdepth, bool gradmode,
                                  AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void Deconvolution3D(unsigned int inchannels, unsigned int outchannels, unsigned int outwidth, unsigned int outheight, unsigned int outdepth,
                                    unsigned int batch, unsigned int kwidth, unsigned int kheight, unsigned int kdepth, bool gradmode,
                                    AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap);

        static void KernelProduct3D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth, unsigned int inheight, unsigned int indepth,
                                    unsigned int batch, unsigned int kwidth, unsigned int kheight, unsigned int kdepth, bool transpose,
                                    AvxArray<float>^ inmap, AvxArray<float>^ outmap, AvxArray<float>^ kernel);
    };
}