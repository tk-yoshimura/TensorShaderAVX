#include "../TensorShaderAvxBackend.h"

#include <limits>

using namespace System;

void averageunpool_1d(unsigned int channels, unsigned int outwidth, unsigned int th, unsigned int stride, float* inmap_ptr, float* outmap_ptr) {
    const unsigned int inwidth = outwidth / stride;
    const unsigned int grad_offset = channels * inwidth * th, map_offset = channels * outwidth * th;
    const unsigned int chsep = channels & ~7, chrem = channels - chsep;

    __m256i mask = TensorShaderAvxBackend::masktable_m256(chrem);
    __m256 inv = _mm256_set1_ps((float)(1.0 / stride));

    inmap_ptr += grad_offset;
    outmap_ptr += map_offset;

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
}

void TensorShaderAvxBackend::Pool::AverageUnpool1D(unsigned int channels, unsigned int outwidth, unsigned int batch, unsigned int th, unsigned int stride, AvxArray<float>^ inmap, AvxArray<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, outmap);

    if (th >= batch) {
        throw gcnew System::ArgumentException();
    }

    unsigned int inwidth = outwidth / stride;

    Util::CheckLength(channels * inwidth * batch, inmap);
    Util::CheckLength(channels * outwidth * batch, outmap);

    outmap->Zeroset(channels * outwidth * th, channels * outwidth);

    float* inmap_ptr = (float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());

    averageunpool_1d(channels, outwidth, th, stride, inmap_ptr, outmap_ptr);
}
