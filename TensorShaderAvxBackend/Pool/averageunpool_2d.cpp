#include "../TensorShaderAvxBackend.h"

#include <limits>

using namespace System;

void averageunpool_2d(unsigned int channels, unsigned int outwidth, unsigned int outheight, unsigned int th, unsigned int stride, float* inmap_ptr, float* outmap_ptr) {
    const unsigned int inwidth = outwidth / stride, inheight = outheight / stride;
    const unsigned int grad_offset = channels * inwidth * inheight * th, map_offset = channels * outwidth * outheight * th;
    const unsigned int chsep = channels & ~7, chrem = channels - chsep;

    __m256i mask = TensorShaderAvxBackend::masktable_m256(chrem);
    __m256 inv = _mm256_set1_ps((float)(1.0 / (stride * stride)));

    inmap_ptr += grad_offset;
    outmap_ptr += map_offset;

    for (unsigned int iy = 0, oy = 0; iy < inheight; iy++, oy += stride) {
        for (unsigned int ix = 0, ox = 0; ix < inwidth; ix++, ox += stride) {
            for (unsigned int ch = 0; ch < chsep; ch += 8) {
                __m256 g = _mm256_loadu_ps(inmap_ptr + ch + (ix + iy * inwidth) * channels);
                __m256 y = _mm256_mul_ps(g, inv);

                for (unsigned int ky = 0; ky < stride; ky++) {
                    for (unsigned int kx = 0; kx < stride; kx++) {
                        _mm256_storeu_ps(outmap_ptr + ch + ((ox + kx) + (oy + ky) * outwidth) * channels, y);
                    }
                }
            }
            if (chrem > 0) {
                __m256 g = _mm256_maskload_ps(inmap_ptr + chsep + (ix + iy * inwidth) * channels, mask);
                __m256 y = _mm256_mul_ps(g, inv);

                for (unsigned int ky = 0; ky < stride; ky++) {
                    for (unsigned int kx = 0; kx < stride; kx++) {
                        _mm256_maskstore_ps(outmap_ptr + chsep + ((ox + kx) + (oy + ky) * outwidth) * channels, mask, y);
                    }
                }
            }
        }
    }
}

void TensorShaderAvxBackend::Pool::AverageUnpool2D(unsigned int channels, unsigned int outwidth, unsigned int outheight, unsigned int batch, unsigned int th, unsigned int stride, AvxArray<float>^ inmap, AvxArray<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, outmap);

    if (th >= batch) {
        throw gcnew System::ArgumentException();
    }

    unsigned int inwidth = outwidth / stride;
    unsigned int inheight = outheight / stride;

    Util::CheckLength(channels * inwidth * inheight * batch, inmap);
    Util::CheckLength(channels * outwidth * outheight * batch, outmap);

    outmap->Zeroset(channels * outwidth * outheight * th, channels * outwidth * outheight);

    float* inmap_ptr = (float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());

    averageunpool_2d(channels, outwidth, outheight, th, stride, inmap_ptr, outmap_ptr);
}
