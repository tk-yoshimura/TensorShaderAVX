#include "../TensorShaderAvxBackend.h"

#include <limits>

using namespace System;

void averageunpool_3d(unsigned int channels, unsigned int outwidth, unsigned int outheight, unsigned int outdepth, unsigned int th, unsigned int stride, float* inmap_ptr, float* outmap_ptr) {
    const unsigned int inwidth = outwidth / stride, inheight = outheight / stride, indepth = outdepth / stride;
    const unsigned int grad_offset = channels * inwidth * inheight * indepth * th, map_offset = channels * outwidth * outheight * outdepth * th;
    const unsigned int chsep = channels & ~7, chrem = channels - chsep;

    __m256i mask = TensorShaderAvxBackend::masktable_m256(chrem);
    __m256 inv = _mm256_set1_ps((float)(1.0 / (stride * stride * stride)));

    inmap_ptr += grad_offset;
    outmap_ptr += map_offset;

    for (unsigned int iz = 0, oz = 0; iz < indepth; iz++, oz += stride) {
        for (unsigned int iy = 0, oy = 0; iy < inheight; iy++, oy += stride) {
            for (unsigned int ix = 0, ox = 0; ix < inwidth; ix++, ox += stride) {
                for (unsigned int ch = 0; ch < chsep; ch += 8) {
                    __m256 g = _mm256_loadu_ps(inmap_ptr + ch + (ix + iy * inwidth + iz * inwidth * inheight) * channels);
                    __m256 y = _mm256_mul_ps(g, inv);

                    for (unsigned int kz = 0; kz < stride; kz++) {
                        for (unsigned int ky = 0; ky < stride; ky++) {
                            for (unsigned int kx = 0; kx < stride; kx++) {
                                _mm256_storeu_ps(outmap_ptr + ch + ((ox + kx) + (oy + ky) * outwidth + (oz + kz) * outwidth * outheight) * channels, y);
                            }
                        }
                    }

                }
                if (chrem > 0) {
                    __m256 g = _mm256_maskload_ps(inmap_ptr + chsep + (ix + iy * inwidth + iz * inwidth * inheight) * channels, mask);
                    __m256 y = _mm256_mul_ps(g, inv);

                    for (unsigned int kz = 0; kz < stride; kz++) {
                        for (unsigned int ky = 0; ky < stride; ky++) {
                            for (unsigned int kx = 0; kx < stride; kx++) {
                                _mm256_maskstore_ps(outmap_ptr + chsep + ((ox + kx) + (oy + ky) * outwidth + (oz + kz) * outwidth * outheight) * channels, mask, y);
                            }
                        }
                    }
                }
            }
        }
    }
}

void TensorShaderAvxBackend::Pool::AverageUnpool3D(unsigned int channels, unsigned int outwidth, unsigned int outheight, unsigned int outdepth, unsigned int batch, unsigned int th, unsigned int stride, AvxArray<float>^ inmap, AvxArray<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, outmap);

    if (th >= batch) {
        throw gcnew System::ArgumentException();
    }

    unsigned int inwidth = outwidth / stride;
    unsigned int inheight = outheight / stride;
    unsigned int indepth = outdepth / stride;

    Util::CheckLength(channels * inwidth * inheight * indepth * batch, inmap);
    Util::CheckLength(channels * outwidth * outheight * outdepth * batch, outmap);

    outmap->Zeroset(channels * outwidth * outheight * outdepth * th, channels * outwidth * outheight * outdepth);
    
    float* inmap_ptr = (float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());

    averageunpool_3d(channels, outwidth, outheight, outdepth, th, stride, inmap_ptr, outmap_ptr);
}
