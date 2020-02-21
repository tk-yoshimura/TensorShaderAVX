#include "../TensorShaderAvxBackend.h"

#include <limits>

using namespace System;

void averageunpool_1d(unsigned int channels, 
                      unsigned int outwidth, 
                      unsigned int batch, unsigned int stride, 
                      const float* __restrict inmap_ptr, float* __restrict outmap_ptr) {
    
    const unsigned int inwidth = outwidth / stride;
    const unsigned int chsep = channels & ~7, chrem = channels - chsep;

    __m256i mask = TensorShaderAvxBackend::masktable_m256(chrem);
    __m256 inv = _mm256_set1_ps((float)(1.0 / stride));

    for (unsigned int th = 0; th < batch; th++) {

        for (unsigned int ix = 0, ox = 0; ix < inwidth; ix++, ox += stride) {
            for (unsigned int ch = 0; ch < chsep; ch += 8) {
                __m256 g = _mm256_loadu_ps(inmap_ptr + ch + ix * channels);
                __m256 y = _mm256_mul_ps(g, inv);

                for (unsigned int kx = 0; kx < stride; kx++) {
                    _mm256_storeu_ps(outmap_ptr + ch + (ox + kx) * channels, y);
                }
            }
            if (chrem > 0) {
                __m256 g = _mm256_maskload_ps(inmap_ptr + chsep + ix * channels, mask);
                __m256 y = _mm256_mul_ps(g, inv);

                for (unsigned int kx = 0; kx < stride; kx++) {
                    _mm256_maskstore_ps(outmap_ptr + chsep + (ox + kx) * channels, mask, y);
                }
            }
        }

        inmap_ptr += channels * inwidth;
        outmap_ptr += channels * outwidth;
    }
}

void TensorShaderAvxBackend::Pool::AverageUnpool1D(unsigned int channels, unsigned int outwidth, unsigned int batch, unsigned int stride, AvxArray<float>^ inmap, AvxArray<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, outmap);

    unsigned int inwidth = outwidth / stride;

    Util::CheckLength(channels * inwidth * batch, inmap);
    Util::CheckLength(channels * outwidth * batch, outmap);

    outmap->Zeroset(channels * outwidth * batch);

    const float* inmap_ptr = (const float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());

    averageunpool_1d(channels, outwidth, batch, stride, inmap_ptr, outmap_ptr);
}
