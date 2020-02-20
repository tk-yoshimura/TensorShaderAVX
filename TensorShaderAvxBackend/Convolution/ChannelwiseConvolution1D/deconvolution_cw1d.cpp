#include "../../TensorShaderAvxBackend.h"

using namespace System;

void deconvolution_cw1d(unsigned int channels, 
                        unsigned inwidth, unsigned outwidth, unsigned kwidth, 
                        unsigned stride, unsigned int th, 
                        const float* __restrict inmap_ptr, float* __restrict outmap_ptr, const float* __restrict kernel_ptr) {
    
    const unsigned int inmap_offset = channels * inwidth * th, outmap_offset = channels * outwidth * th;
    const unsigned int ch_sep = channels & ~7u, ch_rem = channels - ch_sep;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(ch_rem);

    inmap_ptr += inmap_offset;
    outmap_ptr += outmap_offset;

    for (unsigned int ix = 0; ix < inwidth; ix++) {
        for (unsigned int ch = 0; ch < ch_sep; ch += 8) {
            __m256 u = _mm256_loadu_ps(inmap_ptr + ch + channels * ix);
            
            for (unsigned int kx = 0, ox = ix * stride; kx < kwidth; kx++, ox++) {
                __m256 v = _mm256_loadu_ps(kernel_ptr + ch + channels * kx);

                __m256 uv = _mm256_add_ps(_mm256_mul_ps(u, v), _mm256_loadu_ps(outmap_ptr + ch + channels * ox));

                _mm256_storeu_ps(outmap_ptr + ch + channels * ox, uv);
            }
        }

        if (ch_rem > 0) {
            __m256 u = _mm256_maskload_ps(inmap_ptr + ch_sep + channels * ix, mask);

            for (unsigned int kx = 0, ox = ix * stride; kx < kwidth; kx++, ox++) {
                __m256 v = _mm256_maskload_ps(kernel_ptr + ch_sep + channels * kx, mask);

                __m256 uv = _mm256_add_ps(_mm256_mul_ps(u, v), _mm256_maskload_ps(outmap_ptr + ch_sep + channels * ox, mask));

                _mm256_maskstore_ps(outmap_ptr + ch_sep + channels * ox, mask, uv);
            }
        }
    }
}

void TensorShaderAvxBackend::Convolution::ChannelwiseDeconvolution1D(unsigned int channels, unsigned int outwidth,
                                                                     unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int stride,
                                                                     AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, kernel, outmap);
    
    if (th >= batch) {
        throw gcnew System::ArgumentException();
    }

    unsigned int inwidth = (outwidth - kwidth) / stride + 1;

    Util::CheckLength(channels * inwidth * batch, inmap);
    Util::CheckLength(channels * outwidth * batch, outmap);
    Util::CheckLength(channels * kwidth, kernel);

    outmap->Zeroset(channels * outwidth * th, channels * outwidth);
    
    float* inmap_ptr = (float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());
    float* kernel_ptr = (float*)(kernel->Ptr.ToPointer());

    deconvolution_cw1d(channels,
                       inwidth, outwidth, kwidth,
                       stride, th,
                       inmap_ptr, outmap_ptr, kernel_ptr);
}
