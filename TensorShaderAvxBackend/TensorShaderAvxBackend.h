#pragma once

#pragma warning(disable: 4793)

#include <immintrin.h>

using namespace System;

namespace TensorShaderAvxBackend {
    extern __m256i masktable_m256(int k);
    extern __m128i masktable_m128(int k);

    public ref class Util abstract sealed {
        internal:
        static void CheckOutOfRange(unsigned int index, unsigned int length, ... cli::array<cli::array<float>^>^ arrays);
        static void CheckDuplicateArray(... cli::array<cli::array<float>^>^ arrays);
        
        public:
        static property bool IsSupportedAVX { bool get(); };
        static property bool IsSupportedAVX2 { bool get(); };
        static property bool IsSupportedAVX512F { bool get(); };
        static property bool IsSupportedFMA { bool get(); };
    };

    public ref class Elementwise abstract sealed {
        static Elementwise();

        public:
        static void Add(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);
        static void Sub(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);
        static void Mul(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);
        static void Div(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);
        static void Pow(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);
        static void SignedPow(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);

        static void ArcTan2(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);

        static void AddConstant(unsigned int index, unsigned int length, float c, cli::array<float>^ src, cli::array<float>^ dst);
        static void SubConstant(unsigned int index, unsigned int length, float c, cli::array<float>^ src, cli::array<float>^ dst);
        static void MulConstant(unsigned int index, unsigned int length, float c, cli::array<float>^ src, cli::array<float>^ dst);
        static void DivConstant(unsigned int index, unsigned int length, float c, cli::array<float>^ src, cli::array<float>^ dst);
        static void PowConstant(unsigned int index, unsigned int length, float c, cli::array<float>^ src, cli::array<float>^ dst);
        static void SignedPowConstant(unsigned int index, unsigned int length, float c, cli::array<float>^ src, cli::array<float>^ dst);

        static void Maximum(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);
        static void Minimum(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);
        static void Clamp(unsigned int index, unsigned int length, cli::array<float>^ srcval, cli::array<float>^ srcmin, cli::array<float>^ srcmax, cli::array<float>^ dst);

        static void MaximumConstant(unsigned int index, unsigned int length, float c, cli::array<float>^ src, cli::array<float>^ dst);
        static void MinimumConstant(unsigned int index, unsigned int length, float c, cli::array<float>^ src, cli::array<float>^ dst);
        static void ClampConstant(unsigned int index, unsigned int length, float cmin, float cmax, cli::array<float>^ src, cli::array<float>^ dst);
        
        static void GreaterThan(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);
        static void GreaterThanOrEqual(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);

        static void LessThan(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);
        static void LessThanOrEqual(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);

        static void Equal(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);
        static void NotEqual(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);

        static void GreaterThanConstant(unsigned int index, unsigned int length, float c, cli::array<float>^ src, cli::array<float>^ dst);
        static void GreaterThanOrEqualConstant(unsigned int index, unsigned int length, float c, cli::array<float>^ src, cli::array<float>^ dst);

        static void LessThanConstant(unsigned int index, unsigned int length, float c, cli::array<float>^ src, cli::array<float>^ dst);
        static void LessThanOrEqualConstant(unsigned int index, unsigned int length, float c, cli::array<float>^ src, cli::array<float>^ dst);

        static void EqualConstant(unsigned int index, unsigned int length, float c, cli::array<float>^ src, cli::array<float>^ dst);
        static void NotEqualConstant(unsigned int index, unsigned int length, float c, cli::array<float>^ src, cli::array<float>^ dst);

        static void LogicalNot(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        static void LogicalAnd(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);
        static void LogicalOr(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);
        static void LogicalXor(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);

        static void Abs(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        static void Sign(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        static void Step(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        
        static void Neg(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        static void Rcp(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);

        static void Square(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        static void Cube(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);

        static void Sqrt(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        static void Rsqrt(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        static void SignedSqrt(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);

        static void Cbrt(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        static void Rcbrt(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        
        static void Sin(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        static void Cos(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        static void Tan(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);

        static void Sinh(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        static void Cosh(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        static void Tanh(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);

        static void ArcSin(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        static void ArcCos(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        static void ArcTan(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);

        static void Exp(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        static void Exp2(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        
        static void Log(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        static void Log2(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        static void Log10(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);

        static void Floor(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        static void Ceil(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        static void Round(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);

        static void Sigmoid(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        static void SoftPlus(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        static void LogCosh(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);

        static void IsNan(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        static void NanAsZero(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);

        static void Relu(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        static void ReluGrad(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);

        static void LeakyRelu(unsigned int index, unsigned int length, float slope, cli::array<float>^ src, cli::array<float>^ dst);
        static void LeakyReluGrad(unsigned int index, unsigned int length, float slope, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);

        static void Elu(unsigned int index, unsigned int length, float slope, cli::array<float>^ src, cli::array<float>^ dst);
        static void EluGrad(unsigned int index, unsigned int length, float slope, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);

        static void Sum(unsigned int index, unsigned int length, cli::array <cli::array<float>^> ^src, cli::array<float>^ dst);

        static void Lerp(unsigned int index, unsigned int length, cli::array<float>^ srccondition, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);
    };

    public ref class Channelwise abstract sealed {
        static Channelwise();

        public:
        static void Add(unsigned int vector_length, unsigned int map_length, cli::array<float>^ srcvector, cli::array<float>^ srcmap, cli::array<float>^ dstmap);
        static void Mul(unsigned int vector_length, unsigned int map_length, cli::array<float>^ srcvector, cli::array<float>^ srcmap, cli::array<float>^ dstmap);

        static void SubLVector(unsigned int vector_length, unsigned int map_length, cli::array<float>^ srcvector, cli::array<float>^ srcmap, cli::array<float>^ dstmap);
        static void SubRVector(unsigned int vector_length, unsigned int map_length, cli::array<float>^ srcvector, cli::array<float>^ srcmap, cli::array<float>^ dstmap);

        static void DivLVector(unsigned int vector_length, unsigned int map_length, cli::array<float>^ srcvector, cli::array<float>^ srcmap, cli::array<float>^ dstmap);
        static void DivRVector(unsigned int vector_length, unsigned int map_length, cli::array<float>^ srcvector, cli::array<float>^ srcmap, cli::array<float>^ dstmap);
    };

    public ref class Batchwise abstract sealed {
        static Batchwise();

        public:
        static void Mul(unsigned int vector_length, unsigned int map_length, cli::array<float>^ srcvector, cli::array<float>^ srcmap, cli::array<float>^ dstmap);
    };

    public ref class ArrayManipulation abstract sealed {
        static ArrayManipulation();
        
        public:
        static void Zeroset(unsigned int index, unsigned int length, cli::array<float>^ dst);
        static void Clear(unsigned int index, unsigned int length, float c, cli::array<float>^ dst);
        
        static void PatternCopy(unsigned int src_stride, unsigned int src_index, unsigned int dst_stride, unsigned int dst_index, unsigned int copy_length, unsigned int slides, cli::array<float>^ src, cli::array<float>^ dst);
        static void PatternCopy(unsigned int src_offset, unsigned int src_stride, unsigned int src_index, unsigned int dst_offset, unsigned int dst_stride, unsigned int dst_index, unsigned int copy_length, unsigned int slides, cli::array<float>^ src, cli::array<float>^ dst);
                
        static void Broadcast(unsigned int src_index, unsigned int src_length, cli::array<float>^ src, unsigned int dst_index, unsigned int dst_length, cli::array<float>^ dst, unsigned int slides);
        
        static void Flip(unsigned int stride, unsigned int axislength, unsigned int slides, cli::array<float>^ src, cli::array<float>^ dst);
        static void Sort(unsigned int stride, unsigned int axislength, unsigned int slides, cli::array<float>^ src, cli::array<float>^ dst);
        static void SortWithKey(unsigned int stride, unsigned int axislength, unsigned int slides, cli::array<float>^ src_key, cli::array<float>^ src_value, cli::array<float>^ dst_key, cli::array<float>^ dst_value);
    };

    public ref class Indexer abstract sealed {
        static Indexer();

        public:
        static void OneHotVector(unsigned int length, unsigned int channels, cli::array<float>^ src, cli::array<float>^ dst);
        static void ArgMin(unsigned int length, unsigned int channels, cli::array<float>^ src, cli::array<float>^ dst);
        static void ArgMax(unsigned int length, unsigned int channels, cli::array<float>^ src, cli::array<float>^ dst);
        static void Index(unsigned int stride, unsigned int axislength, unsigned int clones, cli::array<float>^ dst);
    };

    public ref class Randomize abstract sealed {
        public:
        static void Uniform(unsigned int index, unsigned int length, cli::array<float>^ dst, Random ^random);
        static void Normal(unsigned int index, unsigned int length, cli::array<float>^ dst, Random^ random);
        static void Bernoulli(unsigned int index, unsigned int length, double prob, cli::array<float>^ dst, Random^ random);
    };

    public ref class Aggregation abstract sealed {
        static Aggregation();

        public:
        static void Sum(unsigned int src_index, unsigned int src_length, cli::array<float>^ src, unsigned int dst_index, unsigned int dst_length, cli::array<float>^ dst, unsigned int slides);
        static void Average(unsigned int src_index, unsigned int src_length, cli::array<float>^ src, unsigned int dst_index, unsigned int dst_length, cli::array<float>^ dst, unsigned int slides);
        static void Min(unsigned int src_index, unsigned int src_length, cli::array<float>^ src, unsigned int dst_index, unsigned int dst_length, cli::array<float>^ dst, unsigned int slides);
        static void Max(unsigned int src_index, unsigned int src_length, cli::array<float>^ src, unsigned int dst_index, unsigned int dst_length, cli::array<float>^ dst, unsigned int slides);
    };

    public ref class Pool abstract sealed {
        static Pool();

        public:
        static void MaxPool1D(unsigned int channels, unsigned int inwidth, unsigned int batch, unsigned int th, unsigned int stride, cli::array<float>^ inmap, cli::array<float>^ outmap);
        static void MaxPool2D(unsigned int channels, unsigned int inwidth, unsigned int inheight, unsigned int batch, unsigned int th, unsigned int stride, cli::array<float>^ inmap, cli::array<float>^ outmap);
        static void MaxPool3D(unsigned int channels, unsigned int inwidth, unsigned int inheight, unsigned int indepth, unsigned int batch, unsigned int th, unsigned int stride, cli::array<float>^ inmap, cli::array<float>^ outmap);

        static void AveragePool1D(unsigned int channels, unsigned int inwidth, unsigned int batch, unsigned int th, unsigned int stride, cli::array<float>^ inmap, cli::array<float>^ outmap);
        static void AveragePool2D(unsigned int channels, unsigned int inwidth, unsigned int inheight, unsigned int batch, unsigned int th, unsigned int stride, cli::array<float>^ inmap, cli::array<float>^ outmap);
        static void AveragePool3D(unsigned int channels, unsigned int inwidth, unsigned int inheight, unsigned int indepth, unsigned int batch, unsigned int th, unsigned int stride, cli::array<float>^ inmap, cli::array<float>^ outmap);

        static void MaxUnpool1D(unsigned int channels, unsigned int outwidth, unsigned int batch, unsigned int th, unsigned int stride, cli::array<float>^ ingrad, cli::array<float>^ inpool, cli::array<float>^ inmap, cli::array<float>^ outmap);
        static void MaxUnpool2D(unsigned int channels, unsigned int outwidth, unsigned int outheight, unsigned int batch, unsigned int th, unsigned int stride, cli::array<float>^ ingrad, cli::array<float>^ inpool, cli::array<float>^ inmap, cli::array<float>^ outmap);
        static void MaxUnpool3D(unsigned int channels, unsigned int outwidth, unsigned int outheight, unsigned int outdepth, unsigned int batch, unsigned int th, unsigned int stride, cli::array<float>^ ingrad, cli::array<float>^ inpool, cli::array<float>^ inmap, cli::array<float>^ outmap);

        static void AverageUnpool1D(unsigned int channels, unsigned int outwidth, unsigned int batch, unsigned int th, unsigned int stride, cli::array<float>^ inmap, cli::array<float>^ outmap);
        static void AverageUnpool2D(unsigned int channels, unsigned int outwidth, unsigned int outheight, unsigned int batch, unsigned int th, unsigned int stride, cli::array<float>^ inmap, cli::array<float>^ outmap);
        static void AverageUnpool3D(unsigned int channels, unsigned int outwidth, unsigned int outheight, unsigned int outdepth, unsigned int batch, unsigned int th, unsigned int stride, cli::array<float>^ inmap, cli::array<float>^ outmap);
    };

    public ref class Transform abstract sealed {
        static Transform();

        public:
        static void ChannelToSpace2D(unsigned int outchannels, unsigned int inwidth, unsigned int inheight, unsigned int batch, unsigned int th, unsigned int scale, cli::array<float>^ inmap, cli::array<float>^ outmap);
        static void SpaceToChannel2D(unsigned int inchannels, unsigned int outwidth, unsigned int outheight, unsigned int batch, unsigned int th, unsigned int scale, cli::array<float>^ inmap, cli::array<float>^ outmap);

        static void ChannelToSpace3D(unsigned int outchannels, unsigned int inwidth, unsigned int inheight, unsigned int indepth, unsigned int batch, unsigned int th, unsigned int scale, cli::array<float>^ inmap, cli::array<float>^ outmap);
        static void SpaceToChannel3D(unsigned int inchannels, unsigned int outwidth, unsigned int outheight, unsigned int outdepth, unsigned int batch, unsigned int th, unsigned int scale, cli::array<float>^ inmap, cli::array<float>^ outmap);
    };

    public ref class Padding abstract sealed {
        static Padding();

        public:
        static void ZeroPadding1D(unsigned int channels, unsigned int inwidth, 
                                  unsigned int batch, unsigned int th, 
                                  unsigned int pad_left, unsigned int pad_right, 
                                  cli::array<float>^ inmap, cli::array<float>^ outmap);

        static void EdgePadding1D(unsigned int channels, unsigned int inwidth, 
                                  unsigned int batch, unsigned int th, 
                                  unsigned int pad_left, unsigned int pad_right, 
                                  cli::array<float>^ inmap, cli::array<float>^ outmap);

        static void ZeroPadding2D(unsigned int channels, unsigned int inwidth, unsigned int inheight, 
                                  unsigned int batch, unsigned int th, 
                                  unsigned int pad_left, unsigned int pad_right, 
                                  unsigned int pad_top, unsigned int pad_bottom,
                                  cli::array<float>^ inmap, cli::array<float>^ outmap);

        static void EdgePadding2D(unsigned int channels, unsigned int inwidth, unsigned int inheight,
                                  unsigned int batch, unsigned int th,
                                  unsigned int pad_left, unsigned int pad_right,
                                  unsigned int pad_top, unsigned int pad_bottom,
                                  cli::array<float>^ inmap, cli::array<float>^ outmap);

        static void ZeroPadding3D(unsigned int channels, unsigned int inwidth, unsigned int inheight, unsigned int indepth,
                                  unsigned int batch, unsigned int th,
                                  unsigned int pad_left, unsigned int pad_right,
                                  unsigned int pad_top, unsigned int pad_bottom,
                                  unsigned int pad_front, unsigned int pad_rear,
                                  cli::array<float>^ inmap, cli::array<float>^ outmap);

        static void EdgePadding3D(unsigned int channels, unsigned int inwidth, unsigned int inheight, unsigned int indepth,
                                  unsigned int batch, unsigned int th,
                                  unsigned int pad_left, unsigned int pad_right,
                                  unsigned int pad_top, unsigned int pad_bottom,
                                  unsigned int pad_front, unsigned int pad_rear,
                                  cli::array<float>^ inmap, cli::array<float>^ outmap);
    };

    public ref class Trimming abstract sealed {
        static Trimming();

        public:
        static void Trimming1D(unsigned int channels, unsigned int outwidth,
                               unsigned int batch, unsigned int th,
                               unsigned int trim_left, unsigned int trim_right,
                               cli::array<float>^ inmap, cli::array<float>^ outmap);

        static void Trimming2D(unsigned int channels, unsigned int outwidth, unsigned int outheight,
                               unsigned int batch, unsigned int th,
                               unsigned int trim_left, unsigned int trim_right,
                               unsigned int trim_top, unsigned int trim_bottom,
                               cli::array<float>^ inmap, cli::array<float>^ outmap);

        static void Trimming3D(unsigned int channels, unsigned int outwidth, unsigned int outheight, unsigned int outdepth,
                               unsigned int batch, unsigned int th,
                               unsigned int trim_left, unsigned int trim_right,
                               unsigned int trim_top, unsigned int trim_bottom,
                               unsigned int trim_front, unsigned int trim_rear,
                               cli::array<float>^ inmap, cli::array<float>^ outmap);
    };

    public ref class Zoom abstract sealed {
        static Zoom();

        public:
        static void NeighborZoom1D(unsigned int channels, unsigned int inwidth,
                                   unsigned int batch, unsigned int th,
                                   cli::array<float>^ inmap, cli::array<float>^ outmap);

        static void NeighborZoom2D(unsigned int channels, unsigned int inwidth, unsigned int inheight,
                                   unsigned int batch, unsigned int th,
                                   cli::array<float>^ inmap, cli::array<float>^ outmap);

        static void NeighborZoom3D(unsigned int channels, unsigned int inwidth, unsigned int inheight, unsigned int indepth,
                                   unsigned int batch, unsigned int th,
                                   cli::array<float>^ inmap, cli::array<float>^ outmap);

        static void LinearZoom1D(unsigned int channels, unsigned int inwidth,
                                 unsigned int batch, unsigned int th,
                                 cli::array<float>^ inmap, cli::array<float>^ outmap);

        static void LinearZoom2D(unsigned int channels, unsigned int inwidth, unsigned int inheight,
                                 unsigned int batch, unsigned int th,
                                 cli::array<float>^ inmap, cli::array<float>^ outmap);

        static void LinearZoom3D(unsigned int channels, unsigned int inwidth, unsigned int inheight, unsigned int indepth,
                                 unsigned int batch, unsigned int th,
                                 cli::array<float>^ inmap, cli::array<float>^ outmap);
    };

    public ref class Convolution abstract sealed {
        static Convolution();

        public:
        static void Dense(unsigned int inchannels, unsigned int outchannels, 
                          unsigned int batch, unsigned int th, 
                          cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void TransposeDense(unsigned int inchannels, unsigned int outchannels, 
                                   unsigned int batch, unsigned int th, 
                                   cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void KernelProductDense(unsigned int inchannels, unsigned int outchannels, 
                                       unsigned int batch, unsigned int outch, 
                                       cli::array<float>^ inmap, cli::array<float>^ outmap, cli::array<float>^ kernel);

        static void Convolution1D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth,
                                  unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int stride,
                                  cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void Deconvolution1D(unsigned int inchannels, unsigned int outchannels, unsigned int outwidth,
                                    unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int stride,
                                    cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);
        
        static void KernelProduct1D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth,
                                    unsigned int batch, unsigned int outch, unsigned int kwidth, unsigned int stride,
                                    cli::array<float>^ inmap, cli::array<float>^ outmap, cli::array<float>^ kernel);

        static void Convolution2D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth, unsigned int inheight,
                                  unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int kheight, unsigned int stride,
                                  cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void Deconvolution2D(unsigned int inchannels, unsigned int outchannels, unsigned int outwidth, unsigned int outheight,
                                    unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int kheight, unsigned int stride,
                                    cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void KernelProduct2D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth, unsigned int inheight,
                                    unsigned int batch, unsigned int outch, unsigned int kwidth, unsigned int kheight, unsigned int stride,
                                    cli::array<float>^ inmap, cli::array<float>^ outmap, cli::array<float>^ kernel);

        static void Convolution3D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth, unsigned int inheight, unsigned int indepth,
                                  unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int kheight, unsigned int kdepth, unsigned int stride,
                                  cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void Deconvolution3D(unsigned int inchannels, unsigned int outchannels, unsigned int outwidth, unsigned int outheight, unsigned int outdepth,
                                    unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int kheight, unsigned int kdepth, unsigned int stride,
                                    cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void KernelProduct3D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth, unsigned int inheight, unsigned int indepth,
                                    unsigned int batch, unsigned int outch, unsigned int kwidth, unsigned int kheight, unsigned int kdepth, unsigned int stride,
                                    cli::array<float>^ inmap, cli::array<float>^ outmap, cli::array<float>^ kernel);

        static void ChannelwiseConvolution1D(unsigned int channels, unsigned int inwidth,
                                             unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int stride,
                                             cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void ChannelwiseDeconvolution1D(unsigned int channels, unsigned int outwidth,
                                               unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int stride,
                                               cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void ChannelwiseKernelProduct1D(unsigned int channels, unsigned int inwidth,
                                               unsigned int batch, unsigned int kwidth, unsigned int stride,
                                               cli::array<float>^ inmap, cli::array<float>^ outmap, cli::array<float>^ kernel);

        static void ChannelwiseConvolution2D(unsigned int channels, unsigned int inwidth, unsigned int inheight,
                                             unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int kheight, unsigned int stride,
                                             cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void ChannelwiseDeconvolution2D(unsigned int channels, unsigned int outwidth, unsigned int outheight,
                                               unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int kheight, unsigned int stride,
                                               cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void ChannelwiseKernelProduct2D(unsigned int channels, unsigned int inwidth, unsigned int inheight,
                                               unsigned int batch, unsigned int kwidth, unsigned int kheight, unsigned int stride,
                                               cli::array<float>^ inmap, cli::array<float>^ outmap, cli::array<float>^ kernel);

        static void ChannelwiseConvolution3D(unsigned int channels, unsigned int inwidth, unsigned int inheight, unsigned int indepth,
                                             unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int kheight, unsigned int kdepth, unsigned int stride,
                                             cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void ChannelwiseDeconvolution3D(unsigned int channels, unsigned int outwidth, unsigned int outheight, unsigned int outdepth,
                                               unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int kheight, unsigned int kdepth, unsigned int stride,
                                               cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void ChannelwiseKernelProduct3D(unsigned int channels, unsigned int inwidth, unsigned int inheight, unsigned int indepth,
                                               unsigned int batch, unsigned int kwidth, unsigned int kheight, unsigned int kdepth, unsigned int stride,
                                               cli::array<float>^ inmap, cli::array<float>^ outmap, cli::array<float>^ kernel);

        static void PointwiseConvolution(unsigned int inchannels, unsigned int outchannels, unsigned int points,
                                         unsigned int batch, unsigned int th, 
                                         cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void PointwiseDeconvolution(unsigned int inchannels, unsigned int outchannels, unsigned int points,
                                           unsigned int batch, unsigned int th, 
                                           cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void PointwiseKernelProduct(unsigned int inchannels, unsigned int outchannels, unsigned int points,
                                           unsigned int batch, unsigned int outch, 
                                           cli::array<float>^ inmap, cli::array<float>^ outmap, cli::array<float>^ kernel);
    };

    public ref class Complex abstract sealed {
        static Complex();

        public:
        static void Mul(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);
        static void MulGrad(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);

        static void RRelu(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        static void RReluGrad(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);

        static void ZRelu(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        static void ZReluGrad(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);

        static void Conjugate(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        
        static void Square(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        static void SquareGrad(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);

        static void Decay(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        static void DecayGrad(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);

        static void Squash(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        static void SquashGrad(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);

        static void Normalize(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        static void NormalizeGrad(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);

        static void Cast(unsigned int length, cli::array<float>^ src_real, cli::array<float>^ src_imag, cli::array<float>^ dst);
        static void Real(unsigned int length, cli::array<float>^ src, cli::array<float>^ dst_real);
        static void Imag(unsigned int length, cli::array<float>^ src, cli::array<float>^ dst_imag);
        static void PureReal(unsigned int length, cli::array<float>^ src_real, cli::array<float>^ dst);
        static void PureImag(unsigned int length, cli::array<float>^ src_imag, cli::array<float>^ dst);

        static void Dense(unsigned int inchannels, unsigned int outchannels,
                          unsigned int batch, unsigned int th, bool gradmode,
                          cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void TransposeDense(unsigned int inchannels, unsigned int outchannels,
                                   unsigned int batch, unsigned int th, bool gradmode,
                                   cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void KernelProductDense(unsigned int inchannels, unsigned int outchannels,
                                       unsigned int batch, unsigned int outch, bool transpose,
                                       cli::array<float>^ inmap, cli::array<float>^ outmap, cli::array<float>^ kernel);

        static void Convolution1D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth,
                                  unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int stride, bool gradmode,
                                  cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void Deconvolution1D(unsigned int inchannels, unsigned int outchannels, unsigned int outwidth,
                                    unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int stride, bool gradmode,
                                    cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void KernelProduct1D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth,
                                    unsigned int batch, unsigned int outch, unsigned int kwidth, unsigned int stride, bool transpose,
                                    cli::array<float>^ inmap, cli::array<float>^ outmap, cli::array<float>^ kernel);

        static void Convolution2D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth, unsigned int inheight,
                                  unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int kheight, unsigned int stride, bool gradmode,
                                  cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void Deconvolution2D(unsigned int inchannels, unsigned int outchannels, unsigned int outwidth, unsigned int outheight,
                                    unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int kheight, unsigned int stride, bool gradmode,
                                    cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void KernelProduct2D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth, unsigned int inheight,
                                    unsigned int batch, unsigned int outch, unsigned int kwidth, unsigned int kheight, unsigned int stride, bool transpose,
                                    cli::array<float>^ inmap, cli::array<float>^ outmap, cli::array<float>^ kernel);

        static void Convolution3D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth, unsigned int inheight, unsigned int indepth,
                                  unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int kheight, unsigned int kdepth, unsigned int stride, bool gradmode,
                                  cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void Deconvolution3D(unsigned int inchannels, unsigned int outchannels, unsigned int outwidth, unsigned int outheight, unsigned int outdepth,
                                    unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int kheight, unsigned int kdepth, unsigned int stride, bool gradmode,
                                    cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void KernelProduct3D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth, unsigned int inheight, unsigned int indepth,
                                    unsigned int batch, unsigned int outch, unsigned int kwidth, unsigned int kheight, unsigned int kdepth, unsigned int stride, bool transpose,
                                    cli::array<float>^ inmap, cli::array<float>^ outmap, cli::array<float>^ kernel);
    };

    public ref class Trivector abstract sealed {
        static Trivector();

        public:
        static void Mul(unsigned int vectorindex, unsigned int quaternionindex, unsigned int vectorlength, cli::array<float>^ src_vector, cli::array<float>^ src_quaternion, cli::array<float>^ dst_vector);
        static void MulQGrad(unsigned int vectorindex, unsigned int quaternionindex, unsigned int vectorlength, cli::array<float>^ src_vector_value, cli::array<float>^ src_vector_grad, cli::array<float>^ src_quaternion, cli::array<float>^ dst_quaternion);
        static void MulVGrad(unsigned int vectorindex, unsigned int quaternionindex, unsigned int vectorlength, cli::array<float>^ src_vector, cli::array<float>^ src_quaternion, cli::array<float>^ dst_vector);

        static void Cross(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);
        
        static void Decay(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        static void DecayGrad(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);

        static void Squash(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        static void SquashGrad(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);

        static void Normalize(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        static void NormalizeGrad(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);

        static void Cast(unsigned int length, cli::array<float>^ src_x, cli::array<float>^ src_y, cli::array<float>^ src_z, cli::array<float>^ dst);
        static void X(unsigned int length, cli::array<float>^ src, cli::array<float>^ dst_x);
        static void Y(unsigned int length, cli::array<float>^ src, cli::array<float>^ dst_y);
        static void Z(unsigned int length, cli::array<float>^ src, cli::array<float>^ dst_z);
        static void PureX(unsigned int length, cli::array<float>^ src_x, cli::array<float>^ dst);
        static void PureY(unsigned int length, cli::array<float>^ src_y, cli::array<float>^ dst);
        static void PureZ(unsigned int length, cli::array<float>^ src_z, cli::array<float>^ dst);

        static void Dense(unsigned int inchannels, unsigned int outchannels,
                          unsigned int batch, unsigned int th, bool gradmode,
                          cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void TransposeDense(unsigned int inchannels, unsigned int outchannels,
                                   unsigned int batch, unsigned int th, bool gradmode,
                                   cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void KernelProductDense(unsigned int inchannels, unsigned int outchannels,
                                       unsigned int batch, unsigned int outch, bool transpose,
                                       cli::array<float>^ inmap, cli::array<float>^ outmap, cli::array<float>^ kernel_value, cli::array<float>^ kernel_grad);


        static void Convolution1D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth,
                                  unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int stride, bool gradmode,
                                  cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void Deconvolution1D(unsigned int inchannels, unsigned int outchannels, unsigned int outwidth,
                                    unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int stride, bool gradmode,
                                    cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void KernelProduct1D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth,
                                    unsigned int batch, unsigned int outch, unsigned int kwidth, unsigned int stride, bool transpose,
                                    cli::array<float>^ inmap, cli::array<float>^ outmap, cli::array<float>^ kernel_value, cli::array<float>^ kernel_grad);

        static void Convolution2D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth, unsigned int inheight,
                                  unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int kheight, unsigned int stride, bool gradmode,
                                  cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void Deconvolution2D(unsigned int inchannels, unsigned int outchannels, unsigned int outwidth, unsigned int outheight,
                                    unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int kheight, unsigned int stride, bool gradmode,
                                    cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void KernelProduct2D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth, unsigned int inheight,
                                    unsigned int batch, unsigned int outch, unsigned int kwidth, unsigned int kheight, unsigned int stride, bool transpose,
                                    cli::array<float>^ inmap, cli::array<float>^ outmap, cli::array<float>^ kernel_value, cli::array<float>^ kernel_grad);

        static void Convolution3D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth, unsigned int inheight, unsigned int indepth,
                                  unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int kheight, unsigned int kdepth, unsigned int stride, bool gradmode,
                                  cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void Deconvolution3D(unsigned int inchannels, unsigned int outchannels, unsigned int outwidth, unsigned int outheight, unsigned int outdepth,
                                    unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int kheight, unsigned int kdepth, unsigned int stride, bool gradmode,
                                    cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void KernelProduct3D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth, unsigned int inheight, unsigned int indepth,
                                    unsigned int batch, unsigned int outch, unsigned int kwidth, unsigned int kheight, unsigned int kdepth, unsigned int stride, bool transpose,
                                    cli::array<float>^ inmap, cli::array<float>^ outmap, cli::array<float>^ kernel_value, cli::array<float>^ kernel_grad);
    };

    public ref class Quaternion abstract sealed {
        static Quaternion();

        public:
        static void Mul(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);
        static void MulGrad(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);
        static void MulTransposeGrad(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);

        static void RRelu(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        static void RReluGrad(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);

        static void Conjugate(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);

        static void Square(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        
        static void Decay(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        static void DecayGrad(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);

        static void Squash(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        static void SquashGrad(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);

        static void Normalize(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst);
        static void NormalizeGrad(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst);

        static void Cast(unsigned int length, cli::array<float>^ src_r, cli::array<float>^ src_i, cli::array<float>^ src_j, cli::array<float>^ src_k, cli::array<float>^ dst);
        static void R(unsigned int length, cli::array<float>^ src, cli::array<float>^ dst_r);
        static void I(unsigned int length, cli::array<float>^ src, cli::array<float>^ dst_i);
        static void J(unsigned int length, cli::array<float>^ src, cli::array<float>^ dst_j);
        static void K(unsigned int length, cli::array<float>^ src, cli::array<float>^ dst_k);
        static void PureR(unsigned int length, cli::array<float>^ src_r, cli::array<float>^ dst);
        static void PureI(unsigned int length, cli::array<float>^ src_i, cli::array<float>^ dst);
        static void PureJ(unsigned int length, cli::array<float>^ src_j, cli::array<float>^ dst);
        static void PureK(unsigned int length, cli::array<float>^ src_k, cli::array<float>^ dst);

        static void Dense(unsigned int inchannels, unsigned int outchannels,
                          unsigned int batch, unsigned int th, bool gradmode,
                          cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void TransposeDense(unsigned int inchannels, unsigned int outchannels,
                                   unsigned int batch, unsigned int th, bool gradmode,
                                   cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void KernelProductDense(unsigned int inchannels, unsigned int outchannels,
                                       unsigned int batch, unsigned int outch, bool transpose,
                                       cli::array<float>^ inmap, cli::array<float>^ outmap, cli::array<float>^ kernel);

        static void Convolution1D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth,
                                  unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int stride, bool gradmode,
                                  cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void Deconvolution1D(unsigned int inchannels, unsigned int outchannels, unsigned int outwidth,
                                    unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int stride, bool gradmode,
                                    cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void KernelProduct1D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth,
                                    unsigned int batch, unsigned int outch, unsigned int kwidth, unsigned int stride, bool transpose,
                                    cli::array<float>^ inmap, cli::array<float>^ outmap, cli::array<float>^ kernel);

        static void Convolution2D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth, unsigned int inheight,
                                  unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int kheight, unsigned int stride, bool gradmode,
                                  cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void Deconvolution2D(unsigned int inchannels, unsigned int outchannels, unsigned int outwidth, unsigned int outheight,
                                    unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int kheight, unsigned int stride, bool gradmode,
                                    cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void KernelProduct2D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth, unsigned int inheight,
                                    unsigned int batch, unsigned int outch, unsigned int kwidth, unsigned int kheight, unsigned int stride, bool transpose,
                                    cli::array<float>^ inmap, cli::array<float>^ outmap, cli::array<float>^ kernel);

        static void Convolution3D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth, unsigned int inheight, unsigned int indepth,
                                  unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int kheight, unsigned int kdepth, unsigned int stride, bool gradmode,
                                  cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void Deconvolution3D(unsigned int inchannels, unsigned int outchannels, unsigned int outwidth, unsigned int outheight, unsigned int outdepth,
                                    unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int kheight, unsigned int kdepth, unsigned int stride, bool gradmode,
                                    cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap);

        static void KernelProduct3D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth, unsigned int inheight, unsigned int indepth,
                                    unsigned int batch, unsigned int outch, unsigned int kwidth, unsigned int kheight, unsigned int kdepth, unsigned int stride, bool transpose,
                                    cli::array<float>^ inmap, cli::array<float>^ outmap, cli::array<float>^ kernel);
    };
}