#include "../TensorShaderAvxBackend.h"

#include <limits>

using namespace System;

void maxunpool_3d(unsigned int channels, unsigned int outwidth, unsigned int outheight, unsigned int outdepth, unsigned int th, unsigned int stride, float* ingrad_ptr, float* inpool_ptr, float* inmap_ptr, float* outmap_ptr) {
    const unsigned int inwidth = outwidth / stride, inheight = outheight / stride, indepth = outdepth / stride;
    const unsigned int grad_offset = channels * inwidth * inheight * indepth * th, map_offset = channels * outwidth * outheight * outdepth * th;
    const unsigned int chsep = channels & ~7, chrem = channels - chsep;

    __m256i mask = TensorShaderAvxBackend::masktable_m256(chrem);

    ingrad_ptr += grad_offset;
    inpool_ptr += grad_offset;
    inmap_ptr += map_offset;
    outmap_ptr += map_offset;

    for (unsigned int iz = 0, oz = 0; iz < indepth; iz++, oz += stride) {
        for (unsigned int iy = 0, oy = 0; iy < inheight; iy++, oy += stride) {
            for (unsigned int ix = 0, ox = 0; ix < inwidth; ix++, ox += stride) {
                for (unsigned int ch = 0; ch < chsep; ch += 8) {
                    __m256 g = _mm256_loadu_ps(ingrad_ptr + ch + (ix + iy * inwidth + iz * inwidth * inheight) * channels);
                    __m256 p = _mm256_loadu_ps(inpool_ptr + ch + (ix + iy * inwidth + iz * inwidth * inheight) * channels);

                    for (unsigned int kz = 0; kz < stride; kz++) {
                        for (unsigned int ky = 0; ky < stride; ky++) {
                            for (unsigned int kx = 0; kx < stride; kx++) {
                                __m256 x = _mm256_loadu_ps(inmap_ptr + ch + ((ox + kx) + (oy + ky) * outwidth + (oz + kz) * outwidth * outheight) * channels);

                                __m256 y = _mm256_and_ps(g, _mm256_cmp_ps(p, x, _CMP_LE_OS));

                                _mm256_storeu_ps(outmap_ptr + ch + ((ox + kx) + (oy + ky) * outwidth + (oz + kz) * outwidth * outheight) * channels, y);
                            }
                        }
                    }

                }
                if (chrem > 0) {
                    __m256 g = _mm256_maskload_ps(ingrad_ptr + chsep + (ix + iy * inwidth + iz * inwidth * inheight) * channels, mask);
                    __m256 p = _mm256_maskload_ps(inpool_ptr + chsep + (ix + iy * inwidth + iz * inwidth * inheight) * channels, mask);

                    for (unsigned int kz = 0; kz < stride; kz++) {
                        for (unsigned int ky = 0; ky < stride; ky++) {
                            for (unsigned int kx = 0; kx < stride; kx++) {
                                __m256 x = _mm256_maskload_ps(inmap_ptr + chsep + ((ox + kx) + (oy + ky) * outwidth + (oz + kz) * outwidth * outheight) * channels, mask);

                                __m256 y = _mm256_and_ps(g, _mm256_cmp_ps(p, x, _CMP_LE_OS));

                                _mm256_maskstore_ps(outmap_ptr + chsep + ((ox + kx) + (oy + ky) * outwidth + (oz + kz) * outwidth * outheight) * channels, mask, y);
                            }
                        }
                    }
                }
            }
        }
    }
}

void TensorShaderAvxBackend::Pool::MaxUnpool3D(unsigned int channels, unsigned int outwidth, unsigned int outheight, unsigned int outdepth, unsigned int batch, unsigned int th, unsigned int stride, cli::array<float>^ ingrad, cli::array<float>^ inpool, cli::array<float>^ inmap, cli::array<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, ingrad, inpool, outmap);

    if (channels * (outwidth / stride) * (outheight / stride) * (outdepth / stride) * batch > (unsigned int)ingrad->Length) {
        throw gcnew System::ArgumentException();
    }
    if (channels * (outwidth / stride) * (outheight / stride) * (outdepth / stride) * batch > (unsigned int)inpool->Length) {
        throw gcnew System::ArgumentException();
    }
    if (channels * outwidth * outheight * outdepth * batch > (unsigned int)inmap->Length) {
        throw gcnew System::ArgumentException();
    }
    if (channels * outwidth * outheight * outdepth * batch > (unsigned int)outmap->Length) {
        throw gcnew System::ArgumentException();
    }
    if (th >= batch) {
        throw gcnew System::ArgumentException();
    }

    ArrayManipulation::Zeroset(channels * outwidth * outheight * outdepth * th, channels * outwidth * outheight * outdepth, outmap);

    pin_ptr<float> pinptr_ingrad = &ingrad[0];
    pin_ptr<float> pinptr_inpool = &inpool[0];
    pin_ptr<float> pinptr_inmap = &inmap[0];
    pin_ptr<float> pinptr_outmap = &outmap[0];

    float* ingrad_ptr = pinptr_ingrad;
    float* inpool_ptr = pinptr_inpool;
    float* inmap_ptr = pinptr_inmap;
    float* outmap_ptr = pinptr_outmap;

    maxunpool_3d(channels, outwidth, outheight, outdepth, th, stride, ingrad_ptr, inpool_ptr, inmap_ptr, outmap_ptr);
}
