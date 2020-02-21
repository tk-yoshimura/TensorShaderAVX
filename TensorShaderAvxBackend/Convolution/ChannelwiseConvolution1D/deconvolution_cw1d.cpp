#include "../../TensorShaderAvxBackend.h"

using namespace System;

void deconvolution_cw1d(unsigned int channels, 
                        unsigned inwidth, unsigned outwidth, unsigned kwidth, 
                        unsigned batch, 
                        const float* __restrict inmap_ptr, float* __restrict outmap_ptr, const float* __restrict kernel_ptr) {
    
    const unsigned int ch_sep = channels & ~7u, ch_rem = channels - ch_sep;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(ch_rem);

    for (unsigned int th = 0; th < batch; th++) {

        for (unsigned int ix = 0; ix < inwidth; ix++) {
            for (unsigned int ch = 0; ch < ch_sep; ch += 8) {
                __m256 u = _mm256_loadu_ps(inmap_ptr + ch + channels * ix);

                for (unsigned int kx = 0, ox = ix; kx < kwidth; kx++, ox++) {
                    __m256 v = _mm256_loadu_ps(kernel_ptr + ch + channels * kx);

                    __m256 uv = _mm256_add_ps(_mm256_mul_ps(u, v), _mm256_loadu_ps(outmap_ptr + ch + channels * ox));

                    _mm256_storeu_ps(outmap_ptr + ch + channels * ox, uv);
                }
            }

            if (ch_rem > 0) {
                __m256 u = _mm256_maskload_ps(inmap_ptr + ch_sep + channels * ix, mask);

                for (unsigned int kx = 0, ox = ix; kx < kwidth; kx++, ox++) {
                    __m256 v = _mm256_maskload_ps(kernel_ptr + ch_sep + channels * kx, mask);

                    __m256 uv = _mm256_add_ps(_mm256_mul_ps(u, v), _mm256_maskload_ps(outmap_ptr + ch_sep + channels * ox, mask));

                    _mm256_maskstore_ps(outmap_ptr + ch_sep + channels * ox, mask, uv);
                }
            }
        }

        inmap_ptr += channels * inwidth;
        outmap_ptr += channels * outwidth;
    }
}

void TensorShaderAvxBackend::Convolution::ChannelwiseDeconvolution1D(unsigned int channels, unsigned int outwidth,
                                                                     unsigned int batch, unsigned int kwidth, 
                                                                     AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, kernel, outmap);

    unsigned int inwidth = outwidth + 1 - kwidth;

    Util::CheckLength(channels * inwidth * batch, inmap);
    Util::CheckLength(channels * outwidth * batch, outmap);
    Util::CheckLength(channels * kwidth, kernel);

    outmap->Zeroset(channels * outwidth * batch);
    
    const float* inmap_ptr = (const float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());
    const float* kernel_ptr = (const float*)(kernel->Ptr.ToPointer());

    deconvolution_cw1d(channels,
                       inwidth, outwidth, kwidth,
                       batch,
                       inmap_ptr, outmap_ptr, kernel_ptr);
}
