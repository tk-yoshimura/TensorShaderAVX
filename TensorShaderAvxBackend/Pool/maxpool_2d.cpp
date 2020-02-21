#include "../TensorShaderAvxBackend.h"

#include <limits>

using namespace System;

void maxpool_2d(unsigned int channels, 
                unsigned int inwidth, unsigned int inheight, 
                unsigned int batch, unsigned int stride, 
                const float* __restrict inmap_ptr, float* __restrict outmap_ptr) {
        
    const unsigned int outwidth = inwidth / stride, outheight = inheight / stride;
    const unsigned int chsep = channels & ~7, chrem = channels - chsep;

    __m256i mask = TensorShaderAvxBackend::masktable_m256(chrem);

    for (unsigned int th = 0; th < batch; th++) {

        for (unsigned int oy = 0, iy = 0; oy < outheight; oy++, iy += stride) {
            for (unsigned int ox = 0, ix = 0; ox < outwidth; ox++, ix += stride) {
                for (unsigned int ch = 0; ch < chsep; ch += 8) {
                    __m256 xmax = _mm256_set1_ps(-HUGE_VALF);

                    for (unsigned int ky = 0; ky < stride; ky++) {
                        for (unsigned int kx = 0; kx < stride; kx++) {
                            __m256 x = _mm256_loadu_ps(inmap_ptr + ch + ((ix + kx) + (iy + ky) * inwidth) * channels);

                            xmax = _mm256_max_ps(x, xmax);
                        }
                    }

                    _mm256_storeu_ps(outmap_ptr + ch + (ox + oy * outwidth) * channels, xmax);
                }
                if (chrem > 0) {
                    __m256 xmax = _mm256_set1_ps(-HUGE_VALF);

                    for (unsigned int ky = 0; ky < stride; ky++) {
                        for (unsigned int kx = 0; kx < stride; kx++) {
                            __m256 x = _mm256_maskload_ps(inmap_ptr + chsep + ((ix + kx) + (iy + ky) * inwidth) * channels, mask);

                            xmax = _mm256_max_ps(x, xmax);
                        }
                    }

                    _mm256_maskstore_ps(outmap_ptr + chsep + (ox + oy * outwidth) * channels, mask, xmax);
                }
            }
        }

        inmap_ptr += channels * inwidth * inheight;
        outmap_ptr += channels * outwidth * outheight;
    }
}

void TensorShaderAvxBackend::Pool::MaxPool2D(unsigned int channels, unsigned int inwidth, unsigned int inheight, unsigned int batch, unsigned int stride, AvxArray<float>^ inmap, AvxArray<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, outmap);

    unsigned int outwidth = inwidth / stride;
    unsigned int outheight = inheight / stride;

    Util::CheckLength(channels * inwidth * inheight * batch, inmap);
    Util::CheckLength(channels * outwidth * outheight * batch, outmap);

    const float* inmap_ptr = (const float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());

    maxpool_2d(channels, inwidth, inheight, batch, stride, inmap_ptr, outmap_ptr);
}
