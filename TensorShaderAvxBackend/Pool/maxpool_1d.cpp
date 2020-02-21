#include "../TensorShaderAvxBackend.h"

using namespace System;

void maxpool_1d(unsigned int channels, 
               unsigned int inwidth, 
               unsigned int batch, unsigned int stride, 
               const float* __restrict inmap_ptr, float* __restrict outmap_ptr) {
        
    const unsigned int outwidth = inwidth / stride;
    const unsigned int chsep = channels & ~7, chrem = channels - chsep; 

    __m256i mask = TensorShaderAvxBackend::masktable_m256(chrem);

    for (unsigned int th = 0; th < batch; th++) {

        for (unsigned int ox = 0, ix = 0; ox < outwidth; ox++, ix += stride) {
            for (unsigned int ch = 0; ch < chsep; ch += 8) {
                __m256 xmax = _mm256_loadu_ps(inmap_ptr + ch + ix * channels);

                for (unsigned int kx = 1; kx < stride; kx++) {
                    __m256 x = _mm256_loadu_ps(inmap_ptr + ch + (ix + kx) * channels);

                    xmax = _mm256_max_ps(x, xmax);
                }

                _mm256_storeu_ps(outmap_ptr + ch + ox * channels, xmax);
            }
            if (chrem > 0) {
                __m256 xmax = _mm256_maskload_ps(inmap_ptr + chsep + ix * channels, mask);

                for (unsigned int kx = 1; kx < stride; kx++) {
                    __m256 x = _mm256_maskload_ps(inmap_ptr + chsep + (ix + kx) * channels, mask);

                    xmax = _mm256_max_ps(x, xmax);
                }

                _mm256_maskstore_ps(outmap_ptr + chsep + ox * channels, mask, xmax);
            }
        }

        inmap_ptr += channels * inwidth;
        outmap_ptr += channels * outwidth;
    }
}

void TensorShaderAvxBackend::Pool::MaxPool1D(unsigned int channels, unsigned int inwidth, unsigned int batch, unsigned int stride, AvxArray<float>^ inmap, AvxArray<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, outmap);

    unsigned int outwidth = inwidth / stride;

    Util::CheckLength(channels * inwidth * batch, inmap);
    Util::CheckLength(channels * outwidth * batch, outmap);

    const float* inmap_ptr = (const float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());

    maxpool_1d(channels, 
               inwidth, 
               batch, stride, 
               inmap_ptr, outmap_ptr);
}
