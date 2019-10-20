#include "../TensorShaderAvxBackend.h"

#include <limits>

using namespace System;

void maxpool_3d(unsigned int channels, unsigned int inwidth, unsigned int inheight, unsigned int indepth, unsigned int th, unsigned int stride, float* inmap_ptr, float* outmap_ptr) {
    const unsigned int outwidth = inwidth / stride, outheight = inheight / stride, outdepth = indepth / stride;
    const unsigned int inmap_offset = channels * inwidth * inheight * indepth * th, outmap_offset = channels * outwidth * outheight * outdepth * th;
    const unsigned int chsep = channels & ~7, chrem = channels - chsep;

    __m256i mask = TensorShaderAvxBackend::masktable_m256(chrem);

    inmap_ptr += inmap_offset;
    outmap_ptr += outmap_offset;

    for (unsigned int oz = 0, iz = 0; oz < outdepth; oz++, iz += stride) {
        for (unsigned int oy = 0, iy = 0; oy < outheight; oy++, iy += stride) {
            for (unsigned int ox = 0, ix = 0; ox < outwidth; ox++, ix += stride) {
                for (unsigned int ch = 0; ch < chsep; ch += 8) {
                    __m256 xmax = _mm256_set1_ps(-HUGE_VALF);

                    for (unsigned int kz = 0; kz < stride; kz++) {
                        for (unsigned int ky = 0; ky < stride; ky++) {
                            for (unsigned int kx = 0; kx < stride; kx++) {
                                __m256 x = _mm256_loadu_ps(inmap_ptr + ch + ((ix + kx) + (iy + ky) * inwidth + (iz + kz) * inwidth * inheight) * channels);

                                xmax = _mm256_max_ps(x, xmax);
                            }
                        }
                    }

                    _mm256_storeu_ps(outmap_ptr + ch + (ox + oy * outwidth + oz * outwidth * outheight) * channels, xmax);
                }
                if (chrem > 0) {
                    __m256 xmax = _mm256_set1_ps(-HUGE_VALF);

                    for (unsigned int kz = 0; kz < stride; kz++) {
                        for (unsigned int ky = 0; ky < stride; ky++) {
                            for (unsigned int kx = 0; kx < stride; kx++) {
                                __m256 x = _mm256_maskload_ps(inmap_ptr + chsep + ((ix + kx) + (iy + ky) * inwidth + (iz + kz) * inwidth * inheight) * channels, mask);

                                xmax = _mm256_max_ps(x, xmax);
                            }
                        }
                    }

                    _mm256_maskstore_ps(outmap_ptr + chsep + (ox + oy * outwidth + oz * outwidth * outheight) * channels, mask, xmax);
                }
            }
        }
    }
}

void TensorShaderAvxBackend::Pool::MaxPool3D(unsigned int channels, unsigned int inwidth, unsigned int inheight, unsigned int indepth, unsigned int batch, unsigned int th, unsigned int stride, cli::array<float>^ inmap, cli::array<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, outmap);

    if (channels * inwidth * inheight * indepth * batch > (unsigned int)inmap->Length) {
        throw gcnew System::ArgumentException();
    }
    if (channels * (inwidth / stride) * (inheight / stride) * (indepth / stride) * batch > (unsigned int)outmap->Length) {
        throw gcnew System::ArgumentException();
    }
    if (th >= batch) {
        throw gcnew System::ArgumentException();
    }

    pin_ptr<float> pinptr_inmap = &inmap[0];
    pin_ptr<float> pinptr_outmap = &outmap[0];

    float* inmap_ptr = pinptr_inmap;
    float* outmap_ptr = pinptr_outmap;

    maxpool_3d(channels, inwidth, inheight, indepth, th, stride, inmap_ptr, outmap_ptr);
}
