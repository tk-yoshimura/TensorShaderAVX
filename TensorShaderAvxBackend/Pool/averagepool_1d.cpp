#include "../TensorShaderAvxBackend.h"

using namespace System;

void broadcastpool_1d(unsigned int channels, unsigned int inwidth, unsigned int th, unsigned int stride, const float* __restrict inmap_ptr, float* outmap_ptr) {
    const unsigned int outwidth = inwidth / stride;
    const unsigned int inmap_offset = channels * inwidth * th, outmap_offset = channels * outwidth * th;
    const unsigned int chsep = channels & ~7, chrem = channels - chsep;

    __m256i mask = TensorShaderAvxBackend::masktable_m256(chrem);
    __m256 inv = _mm256_set1_ps((float)(1.0 / (stride)));

    inmap_ptr += inmap_offset;
    outmap_ptr += outmap_offset;

    for (unsigned int ox = 0, ix = 0; ox < outwidth; ox++, ix += stride) {
        for (unsigned int ch = 0; ch < chsep; ch += 8) {
            __m256 xsum = _mm256_loadu_ps(inmap_ptr + ch + ix * channels);

            for (unsigned int kx = 1; kx < stride; kx++) {
                __m256 x = _mm256_loadu_ps(inmap_ptr + ch + (ix + kx) * channels);

                xsum = _mm256_add_ps(x, xsum);
            }

            __m256 xavg = _mm256_mul_ps(xsum, inv);

            _mm256_storeu_ps(outmap_ptr + ch + ox * channels, xavg);
        }
        if (chrem > 0) {
            __m256 xsum = _mm256_maskload_ps(inmap_ptr + chsep + ix * channels, mask);

            for (unsigned int kx = 1; kx < stride; kx++) {
                __m256 x = _mm256_maskload_ps(inmap_ptr + chsep + (ix + kx) * channels, mask);

                xsum = _mm256_add_ps(x, xsum);
            }

            __m256 xavg = _mm256_mul_ps(xsum, inv);

            _mm256_maskstore_ps(outmap_ptr + chsep + ox * channels, mask, xavg);
        }
    }
}

void TensorShaderAvxBackend::Pool::AveragePool1D(unsigned int channels, unsigned int inwidth, unsigned int batch, unsigned int th, unsigned int stride, AvxArray<float>^ inmap, AvxArray<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, outmap);

    if (th >= batch) {
        throw gcnew System::ArgumentException();
    }

    unsigned int outwidth = inwidth / stride;

    Util::CheckLength(channels * inwidth * batch, inmap);
    Util::CheckLength(channels * outwidth * batch, outmap);

    float* inmap_ptr = (float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());

    broadcastpool_1d(channels, inwidth, th, stride, inmap_ptr, outmap_ptr);
}
