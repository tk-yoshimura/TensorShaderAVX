#include "../../TensorShaderAvxBackend.h"

using namespace System;

void kernelproduct_cw3d(unsigned int channels,
                        unsigned inwidth, unsigned outwidth, unsigned kwidth,
                        unsigned inheight, unsigned outheight, unsigned kheight,
                        unsigned indepth, unsigned outdepth, unsigned kdepth,
                        unsigned stride, unsigned batch,
                        float* inmap_ptr, float* outmap_ptr, float* kernel_ptr) {

    const unsigned int ch_sep = channels & ~7u, ch_rem = channels - ch_sep;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(ch_rem);

    for (unsigned int ch = 0; ch < ch_sep; ch += 8) {

        for (unsigned int kz = 0; kz < kdepth; kz++) {
            for (unsigned int ky = 0; ky < kheight; ky++) {
                for (unsigned int kx = 0; kx < kwidth; kx++) {
                    __m256d uv_hi = _mm256_setzero_pd(), uv_lo = _mm256_setzero_pd();

                    for (unsigned int th = 0; th < batch; th++) {
                        for (unsigned int oz = 0, iz = kz; oz < outdepth; oz++, iz += stride) {
                            for (unsigned int oy = 0, iy = ky; oy < outheight; oy++, iy += stride) {
                                for (unsigned int ox = 0, ix = kx; ox < outwidth; ox++, ix += stride) {

                                    __m256 u = _mm256_loadu_ps(inmap_ptr + ch + channels * (ix + inwidth * (iy + inheight * (iz + indepth * th))));
                                    __m256 v = _mm256_loadu_ps(outmap_ptr + ch + channels * (ox + outwidth * (oy + outheight * (oz + outdepth * th))));

                                    __m256d u_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(u, 1));
                                    __m256d u_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(u));

                                    __m256d v_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1));
                                    __m256d v_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(v));

                                    uv_hi = _mm256_fmadd_pd(u_hi, v_hi, uv_hi);
                                    uv_lo = _mm256_fmadd_pd(u_lo, v_lo, uv_lo);
                                }
                            }
                        }
                    }

                    _mm_storeu_ps(kernel_ptr + ch + channels * (kx + kwidth * (ky + kheight * kz)), _mm256_cvtpd_ps(uv_lo));
                    _mm_storeu_ps(kernel_ptr + ch + channels * (kx + kwidth * (ky + kheight * kz)) + 4, _mm256_cvtpd_ps(uv_hi));
                }
            }
        }
    }

    if (ch_rem > 0) {

        for (unsigned int kz = 0; kz < kdepth; kz++) {
            for (unsigned int ky = 0; ky < kheight; ky++) {
                for (unsigned int kx = 0; kx < kwidth; kx++) {
                    __m256d uv_hi = _mm256_setzero_pd(), uv_lo = _mm256_setzero_pd();

                    for (unsigned int th = 0; th < batch; th++) {
                        for (unsigned int oz = 0, iz = kz; oz < outdepth; oz++, iz += stride) {
                            for (unsigned int oy = 0, iy = ky; oy < outheight; oy++, iy += stride) {
                                for (unsigned int ox = 0, ix = kx; ox < outwidth; ox++, ix += stride) {

                                    __m256 u = _mm256_maskload_ps(inmap_ptr + ch_sep + channels * (ix + inwidth * (iy + inheight * (iz + indepth * th))), mask);
                                    __m256 v = _mm256_maskload_ps(outmap_ptr + ch_sep + channels * (ox + outwidth * (oy + outheight * (oz + outdepth * th))), mask);

                                    __m256d u_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(u, 1));
                                    __m256d u_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(u));

                                    __m256d v_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1));
                                    __m256d v_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(v));

                                    uv_hi = _mm256_fmadd_pd(u_hi, v_hi, uv_hi);
                                    uv_lo = _mm256_fmadd_pd(u_lo, v_lo, uv_lo);
                                }
                            }
                        }
                    }

                    if (ch_rem > 4) {
                        _mm_storeu_ps(kernel_ptr + ch_sep + channels * (kx + kwidth * (ky + kheight * kz)), _mm256_cvtpd_ps(uv_lo));
                        _mm_maskstore_ps(kernel_ptr + ch_sep + channels * (kx + kwidth * (ky + kheight * kz)) + 4, TensorShaderAvxBackend::masktable_m128(ch_rem - 4), _mm256_cvtpd_ps(uv_hi));
                    }
                    else if (ch_rem >= 4) {
                        _mm_storeu_ps(kernel_ptr + ch_sep + channels * (kx + kwidth * (ky + kheight * kz)), _mm256_cvtpd_ps(uv_lo));
                    }
                    else {
                        _mm_maskstore_ps(kernel_ptr + ch_sep + channels * (kx + kwidth * (ky + kheight * kz)), TensorShaderAvxBackend::masktable_m128(ch_rem), _mm256_cvtpd_ps(uv_lo));
                    }
                }
            }
        }
    }
}

void TensorShaderAvxBackend::Convolution::ChannelwiseKernelProduct3D(unsigned int channels, unsigned int inwidth, unsigned int inheight, unsigned int indepth,
                                                                     unsigned int batch, unsigned int kwidth, unsigned int kheight, unsigned int kdepth, unsigned int stride,
                                                                     AvxArray<float>^ inmap, AvxArray<float>^ outmap, AvxArray<float>^ kernel) {

    Util::CheckDuplicateArray(inmap, kernel, outmap);

    unsigned int outwidth = (inwidth - kwidth) / stride + 1;
    unsigned int outheight = (inheight - kheight) / stride + 1;
    unsigned int outdepth = (indepth - kdepth) / stride + 1;

    Util::CheckLength(channels * inwidth * inheight * indepth * batch, inmap);
    Util::CheckLength(channels * outwidth * outheight * outdepth * batch, outmap);
    Util::CheckLength(channels * kwidth * kheight * kdepth, kernel);

    float* inmap_ptr = (float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());
    float* kernel_ptr = (float*)(kernel->Ptr.ToPointer());

    kernelproduct_cw3d(channels, 
                       inwidth, outwidth, kwidth, 
                       inheight, outheight, kheight,
                       indepth, outdepth, kdepth,
                       stride, batch, 
                       inmap_ptr, outmap_ptr, kernel_ptr);
}
