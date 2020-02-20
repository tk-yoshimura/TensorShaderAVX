#include "../TensorShaderAvxBackend.h"

using namespace System;

void broadcastpool_2d(unsigned int channels, unsigned int inwidth, unsigned int inheight, unsigned int th, unsigned int stride, const float* __restrict inmap_ptr, float* outmap_ptr) {
    const unsigned int outwidth = inwidth / stride, outheight = inheight / stride;
    const unsigned int inmap_offset = channels * inwidth * inheight * th, outmap_offset = channels * outwidth * outheight * th;
    const unsigned int chsep = channels & ~7, chrem = channels - chsep;

    __m256i mask = TensorShaderAvxBackend::masktable_m256(chrem);
    __m256 inv = _mm256_set1_ps((float)(1.0 / (stride * stride)));

    inmap_ptr += inmap_offset;
    outmap_ptr += outmap_offset;

    for (unsigned int oy = 0, iy = 0; oy < outheight; oy++, iy += stride) {
        for (unsigned int ox = 0, ix = 0; ox < outwidth; ox++, ix += stride) {
            for (unsigned int ch = 0; ch < chsep; ch += 8) {
                __m256 xsum = _mm256_setzero_ps();

                for (unsigned int ky = 0; ky < stride; ky++) {
                    for (unsigned int kx = 0; kx < stride; kx++) {
                        __m256 x = _mm256_loadu_ps(inmap_ptr + ch + ((ix + kx) + (iy + ky) * inwidth) * channels);

                        xsum = _mm256_add_ps(x, xsum);
                    }
                }

                __m256 xavg = _mm256_mul_ps(xsum, inv);

                _mm256_storeu_ps(outmap_ptr + ch + (ox + oy * outwidth) * channels, xavg);
            }
            if (chrem > 0) {
                __m256 xsum = _mm256_setzero_ps();

                for (unsigned int ky = 0; ky < stride; ky++) {
                    for (unsigned int kx = 0; kx < stride; kx++) {
                        __m256 x = _mm256_maskload_ps(inmap_ptr + chsep + ((ix + kx) + (iy + ky) * inwidth) * channels, mask);

                        xsum = _mm256_add_ps(x, xsum);
                    }
                }

                __m256 xavg = _mm256_mul_ps(xsum, inv);

                _mm256_maskstore_ps(outmap_ptr + chsep + (ox + oy * outwidth) * channels, mask, xavg);
            }
        }
    }
}

void TensorShaderAvxBackend::Pool::AveragePool2D(unsigned int channels, unsigned int inwidth, unsigned int inheight, unsigned int batch, unsigned int th, unsigned int stride, AvxArray<float>^ inmap, AvxArray<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, outmap);

    if (th >= batch) {
        throw gcnew System::ArgumentException();
    }

    unsigned int outwidth = inwidth / stride;
    unsigned int outheight = inheight / stride;

    Util::CheckLength(channels * inwidth * inheight * batch, inmap);
    Util::CheckLength(channels * outwidth * outheight * batch, outmap);

    float* inmap_ptr = (float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());

    broadcastpool_2d(channels, inwidth, inheight, th, stride, inmap_ptr, outmap_ptr);
}
