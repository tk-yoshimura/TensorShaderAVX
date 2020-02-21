#include "../../TensorShaderAvxBackend.h"

using namespace System;

void deconvolution_cw2d(unsigned int channels,
                        unsigned inwidth, unsigned outwidth, unsigned kwidth, 
                        unsigned inheight, unsigned outheight, unsigned kheight,
                        unsigned stride, unsigned int th, 
                        const float* __restrict inmap_ptr, float* __restrict outmap_ptr, const float* __restrict kernel_ptr) {
    
    const unsigned int inmap_offset = channels * inwidth * inheight * th, outmap_offset = channels * outwidth * outheight * th;
    const unsigned int ch_sep = channels & ~7u, ch_rem = channels - ch_sep;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(ch_rem);

    inmap_ptr += inmap_offset;
    outmap_ptr += outmap_offset;

    for (unsigned int iy = 0; iy < inheight; iy++) {
        for (unsigned int ix = 0; ix < inwidth; ix++) {
            for (unsigned int ch = 0; ch < ch_sep; ch += 8) {
                __m256 u = _mm256_loadu_ps(inmap_ptr + ch + channels * (ix + inwidth * iy));

                for (unsigned int ky = 0, oy = iy * stride; ky < kheight; ky++, oy++) {
                    for (unsigned int kx = 0, ox = ix * stride; kx < kwidth; kx++, ox++) {
                        __m256 v = _mm256_loadu_ps(kernel_ptr + ch + channels * (kx + kwidth * ky));

                        __m256 uv = _mm256_add_ps(_mm256_mul_ps(u, v), _mm256_loadu_ps(outmap_ptr + ch + channels * (ox + outwidth * oy)));

                        _mm256_storeu_ps(outmap_ptr + ch + channels * (ox + outwidth * oy), uv);
                    }
                }
            }

            if (ch_rem > 0) {
                __m256 u = _mm256_maskload_ps(inmap_ptr + ch_sep + channels * (ix + inwidth * iy), mask);

                for (unsigned int ky = 0, oy = iy * stride; ky < kheight; ky++, oy++) {
                    for (unsigned int kx = 0, ox = ix * stride; kx < kwidth; kx++, ox++) {
                        __m256 v = _mm256_maskload_ps(kernel_ptr + ch_sep + channels * (kx + kwidth * ky), mask);

                        __m256 uv = _mm256_add_ps(_mm256_mul_ps(u, v), _mm256_maskload_ps(outmap_ptr + ch_sep + channels * (ox + outwidth * oy), mask));

                        _mm256_maskstore_ps(outmap_ptr + ch_sep + channels * (ox + outwidth * oy), mask, uv);
                    }
                }
            }
        }
    }
}

void TensorShaderAvxBackend::Convolution::ChannelwiseDeconvolution2D(unsigned int channels, unsigned int outwidth, unsigned int outheight,
                                                                     unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int kheight, unsigned int stride,
                                                                     AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, kernel, outmap);

    if (th >= batch) {
        throw gcnew System::ArgumentException();
    }

    unsigned int inwidth = outwidth + 1 - kwidth;
    unsigned int inheight = outheight + 1 - kheight;

    Util::CheckLength(channels * inwidth * inheight * batch, inmap);
    Util::CheckLength(channels * outwidth * outheight * batch, outmap);
    Util::CheckLength(channels * kwidth * kheight, kernel);

    outmap->Zeroset(channels * outwidth * outheight * th, channels * outwidth * outheight);

    const float* inmap_ptr = (const float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());
    float* kernel_ptr = (float*)(kernel->Ptr.ToPointer());

    deconvolution_cw2d(channels,
                       inwidth, outwidth, kwidth,
                       inheight, outheight, kheight,
                       stride, th,
                       inmap_ptr, outmap_ptr, kernel_ptr);
}
