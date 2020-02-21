#include "../TensorShaderAvxBackend.h"

#include <limits>

using namespace System;

void maxunpool_1d(unsigned int channels, 
                  unsigned int outwidth, 
                  unsigned int batch, unsigned int stride, 
                  const float* __restrict ingrad_ptr, const float* __restrict inpool_ptr, const float* __restrict inmap_ptr, float* __restrict outmap_ptr) {
    
    const unsigned int inwidth = outwidth / stride;
    const unsigned int chsep = channels & ~7, chrem = channels - chsep;

    __m256i mask = TensorShaderAvxBackend::masktable_m256(chrem);

    for (unsigned int th = 0; th < batch; th++) {

        for (unsigned int ix = 0, ox = 0; ix < inwidth; ix++, ox += stride) {
            for (unsigned int ch = 0; ch < chsep; ch += 8) {
                __m256 g = _mm256_loadu_ps(ingrad_ptr + ch + ix * channels);
                __m256 p = _mm256_loadu_ps(inpool_ptr + ch + ix * channels);

                for (unsigned int kx = 0; kx < stride; kx++) {
                    __m256 x = _mm256_loadu_ps(inmap_ptr + ch + (ox + kx) * channels);

                    __m256 y = _mm256_and_ps(g, _mm256_cmp_ps(p, x, _CMP_LE_OS));

                    _mm256_storeu_ps(outmap_ptr + ch + (ox + kx) * channels, y);
                }

            }
            if (chrem > 0) {
                __m256 g = _mm256_maskload_ps(ingrad_ptr + chsep + ix * channels, mask);
                __m256 p = _mm256_maskload_ps(inpool_ptr + chsep + ix * channels, mask);

                for (unsigned int kx = 0; kx < stride; kx++) {
                    __m256 x = _mm256_maskload_ps(inmap_ptr + chsep + (ox + kx) * channels, mask);

                    __m256 y = _mm256_and_ps(g, _mm256_cmp_ps(p, x, _CMP_LE_OS));

                    _mm256_maskstore_ps(outmap_ptr + chsep + (ox + kx) * channels, mask, y);
                }
            }
        }

        ingrad_ptr += channels * inwidth;
        inpool_ptr += channels * inwidth;
        inmap_ptr += channels * outwidth;
        outmap_ptr += channels * outwidth;
    }
}

void TensorShaderAvxBackend::Pool::MaxUnpool1D(unsigned int channels, unsigned int outwidth, unsigned int batch, unsigned int stride, AvxArray<float>^ ingrad, AvxArray<float>^ inpool, AvxArray<float>^ inmap, AvxArray<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, ingrad, inpool, outmap);

    unsigned int inwidth = outwidth / stride;

    Util::CheckLength(channels * inwidth * batch, ingrad, inpool);
    Util::CheckLength(channels * outwidth * batch, inmap, outmap);

    outmap->Zeroset(channels * outwidth * batch);

    const float* ingrad_ptr = (const float*)(ingrad->Ptr.ToPointer());
    const float* inpool_ptr = (const float*)(inpool->Ptr.ToPointer());
    const float* inmap_ptr = (const float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());

    maxunpool_1d(channels, outwidth, batch, stride, ingrad_ptr, inpool_ptr, inmap_ptr, outmap_ptr);
}
