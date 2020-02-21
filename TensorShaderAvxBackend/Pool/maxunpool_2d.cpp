#include "../TensorShaderAvxBackend.h"

#include <limits>

using namespace System;

void maxunpool_2d(unsigned int channels, 
                  unsigned int outwidth, unsigned int outheight, 
                  unsigned int batch, unsigned int stride, 
                  const float* __restrict ingrad_ptr, const float* __restrict inpool_ptr, const float* __restrict inmap_ptr, float* __restrict outmap_ptr) {
    
    const unsigned int inwidth = outwidth / stride, inheight = outheight / stride;
    const unsigned int chsep = channels & ~7, chrem = channels - chsep;

    __m256i mask = TensorShaderAvxBackend::masktable_m256(chrem);

    for (unsigned int th = 0; th < batch; th++) {

        for (unsigned int iy = 0, oy = 0; iy < inheight; iy++, oy += stride) {
            for (unsigned int ix = 0, ox = 0; ix < inwidth; ix++, ox += stride) {
                for (unsigned int ch = 0; ch < chsep; ch += 8) {
                    __m256 g = _mm256_loadu_ps(ingrad_ptr + ch + (ix + iy * inwidth) * channels);
                    __m256 p = _mm256_loadu_ps(inpool_ptr + ch + (ix + iy * inwidth) * channels);

                    for (unsigned int ky = 0; ky < stride; ky++) {
                        for (unsigned int kx = 0; kx < stride; kx++) {
                            __m256 x = _mm256_loadu_ps(inmap_ptr + ch + ((ox + kx) + (oy + ky) * outwidth) * channels);

                            __m256 y = _mm256_and_ps(g, _mm256_cmp_ps(p, x, _CMP_LE_OS));

                            _mm256_storeu_ps(outmap_ptr + ch + ((ox + kx) + (oy + ky) * outwidth) * channels, y);
                        }
                    }

                }
                if (chrem > 0) {
                    __m256 g = _mm256_maskload_ps(ingrad_ptr + chsep + (ix + iy * inwidth) * channels, mask);
                    __m256 p = _mm256_maskload_ps(inpool_ptr + chsep + (ix + iy * inwidth) * channels, mask);

                    for (unsigned int ky = 0; ky < stride; ky++) {
                        for (unsigned int kx = 0; kx < stride; kx++) {
                            __m256 x = _mm256_maskload_ps(inmap_ptr + chsep + ((ox + kx) + (oy + ky) * outwidth) * channels, mask);

                            __m256 y = _mm256_and_ps(g, _mm256_cmp_ps(p, x, _CMP_LE_OS));

                            _mm256_maskstore_ps(outmap_ptr + chsep + ((ox + kx) + (oy + ky) * outwidth) * channels, mask, y);
                        }
                    }
                }
            }
        }

        ingrad_ptr += channels * inwidth * inheight;
        inpool_ptr += channels * inwidth * inheight;
        inmap_ptr += channels * outwidth * outheight;
        outmap_ptr += channels * outwidth * outheight;
    }
}

void TensorShaderAvxBackend::Pool::MaxUnpool2D(unsigned int channels, unsigned int outwidth, unsigned int outheight, unsigned int batch, unsigned int stride, AvxArray<float>^ ingrad, AvxArray<float>^ inpool, AvxArray<float>^ inmap, AvxArray<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, ingrad, inpool, outmap);

    unsigned int inwidth = outwidth / stride;
    unsigned int inheight = outheight / stride;

    Util::CheckLength(channels * inwidth * inheight * batch, ingrad, inpool);
    Util::CheckLength(channels * outwidth * outheight * batch, inmap, outmap);

    outmap->Zeroset(channels * outwidth * outheight * batch);

    const float* ingrad_ptr = (const float*)(ingrad->Ptr.ToPointer());
    const float* inpool_ptr = (const float*)(inpool->Ptr.ToPointer());
    const float* inmap_ptr = (const float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());

    maxunpool_2d(channels, outwidth, outheight, batch, stride, ingrad_ptr, inpool_ptr, inmap_ptr, outmap_ptr);
}
