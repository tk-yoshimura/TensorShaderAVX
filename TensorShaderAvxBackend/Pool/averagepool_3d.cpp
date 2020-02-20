#include "../TensorShaderAvxBackend.h"

using namespace System;

void broadcastpool_3d(unsigned int channels, unsigned int inwidth, unsigned int inheight, unsigned int indepth, unsigned int th, unsigned int stride, const float* __restrict inmap_ptr, float* outmap_ptr) {
    const unsigned int outwidth = inwidth / stride, outheight = inheight / stride, outdepth = indepth / stride;
    const unsigned int inmap_offset = channels * inwidth * inheight * indepth * th, outmap_offset = channels * outwidth * outheight * outdepth * th;
    const unsigned int chsep = channels & ~7, chrem = channels - chsep;

    __m256i mask = TensorShaderAvxBackend::masktable_m256(chrem);
    __m256 inv = _mm256_set1_ps((float)(1.0 / (stride * stride * stride)));

    inmap_ptr += inmap_offset;
    outmap_ptr += outmap_offset;

    for (unsigned int oz = 0, iz = 0; oz < outdepth; oz++, iz += stride) {
        for (unsigned int oy = 0, iy = 0; oy < outheight; oy++, iy += stride) {
            for (unsigned int ox = 0, ix = 0; ox < outwidth; ox++, ix += stride) {
                for (unsigned int ch = 0; ch < chsep; ch += 8) {
                    __m256 xsum = _mm256_setzero_ps();

                    for (unsigned int kz = 0; kz < stride; kz++) {
                        for (unsigned int ky = 0; ky < stride; ky++) {
                            for (unsigned int kx = 0; kx < stride; kx++) {
                                __m256 x = _mm256_loadu_ps(inmap_ptr + ch + ((ix + kx) + (iy + ky) * inwidth + (iz + kz) * inwidth * inheight) * channels);

                                xsum = _mm256_add_ps(x, xsum);
                            }
                        }
                    }

                    __m256 xavg = _mm256_mul_ps(xsum, inv);

                    _mm256_storeu_ps(outmap_ptr + ch + (ox + oy * outwidth + oz * outwidth * outheight) * channels, xavg);
                }
                if (chrem > 0) {
                    __m256 xsum = _mm256_setzero_ps();

                    for (unsigned int kz = 0; kz < stride; kz++) {
                        for (unsigned int ky = 0; ky < stride; ky++) {
                            for (unsigned int kx = 0; kx < stride; kx++) {
                                __m256 x = _mm256_maskload_ps(inmap_ptr + chsep + ((ix + kx) + (iy + ky) * inwidth + (iz + kz) * inwidth * inheight) * channels, mask);

                                xsum = _mm256_add_ps(x, xsum);
                            }
                        }
                    }

                    __m256 xavg = _mm256_mul_ps(xsum, inv);

                    _mm256_maskstore_ps(outmap_ptr + chsep + (ox + oy * outwidth + oz * outwidth * outheight) * channels, mask, xavg);
                }
            }
        }
    }
}

void TensorShaderAvxBackend::Pool::AveragePool3D(unsigned int channels, unsigned int inwidth, unsigned int inheight, unsigned int indepth, unsigned int batch, unsigned int th, unsigned int stride, AvxArray<float>^ inmap, AvxArray<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, outmap);

    if (th >= batch) {
        throw gcnew System::ArgumentException();
    }

    unsigned int outwidth = inwidth / stride;
    unsigned int outheight = inheight / stride;
    unsigned int outdepth = indepth / stride;

    Util::CheckLength(channels * inwidth * inheight * indepth * batch, inmap);
    Util::CheckLength(channels * outwidth * outheight * outdepth * batch, outmap);

    float* inmap_ptr = (float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());

    broadcastpool_3d(channels, inwidth, inheight, indepth, th, stride, inmap_ptr, outmap_ptr);
}
