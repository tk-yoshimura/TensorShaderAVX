#include "../../TensorShaderAvxBackend.h"

using namespace System;

void kernelproduct_cw1d(unsigned int channels,
                      unsigned inwidth, unsigned outwidth, unsigned kwidth,
                      unsigned stride, unsigned batch, 
                      float* inmap_ptr, float* outmap_ptr, float* kernel_ptr) {

    const unsigned int ch_sep = channels & ~7u, ch_rem = channels - ch_sep;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(ch_rem);
    
    for (unsigned int ch = 0; ch < ch_sep; ch += 8) {

        for (unsigned int kx = 0; kx < kwidth; kx++) {
            __m256d uv_hi = _mm256_setzero_pd(), uv_lo = _mm256_setzero_pd();

            for (unsigned int th = 0; th < batch; th++) {
                for (unsigned int ox = 0, ix = kx; ox < outwidth; ox++, ix += stride) {

                    __m256 u = _mm256_loadu_ps(inmap_ptr + ch + channels * (ix + inwidth * th));
                    __m256 v = _mm256_loadu_ps(outmap_ptr + ch + channels * (ox + outwidth * th));

                    __m256d u_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(u, 1));
                    __m256d u_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(u));

                    __m256d v_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1));
                    __m256d v_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(v));

                    uv_hi = _mm256_fmadd_pd(u_hi, v_hi, uv_hi);
                    uv_lo = _mm256_fmadd_pd(u_lo, v_lo, uv_lo);
                }
            }

            _mm_storeu_ps(kernel_ptr + ch + channels * kx, _mm256_cvtpd_ps(uv_lo));
            _mm_storeu_ps(kernel_ptr + ch + channels * kx + 4, _mm256_cvtpd_ps(uv_hi));
        }
    }

    if (ch_rem > 0) {

        for (unsigned int kx = 0; kx < kwidth; kx++) {
            __m256d uv_hi = _mm256_setzero_pd(), uv_lo = _mm256_setzero_pd();

            for (unsigned int th = 0; th < batch; th++) {
                for (unsigned int ox = 0, ix = kx; ox < outwidth; ox++, ix += stride) {

                    __m256 u = _mm256_maskload_ps(inmap_ptr + ch_sep + channels * (ix + inwidth * th), mask);
                    __m256 v = _mm256_maskload_ps(outmap_ptr + ch_sep + channels * (ox + outwidth * th), mask);

                    __m256d u_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(u, 1));
                    __m256d u_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(u));

                    __m256d v_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1));
                    __m256d v_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(v));

                    uv_hi = _mm256_fmadd_pd(u_hi, v_hi, uv_hi);
                    uv_lo = _mm256_fmadd_pd(u_lo, v_lo, uv_lo);
                }
            }

            if (ch_rem > 4) {
                _mm_storeu_ps(kernel_ptr + ch_sep + channels * kx, _mm256_cvtpd_ps(uv_lo));
                _mm_maskstore_ps(kernel_ptr + ch_sep + channels * kx + 4, TensorShaderAvxBackend::masktable_m128(ch_rem - 4), _mm256_cvtpd_ps(uv_hi));
            }
            else if (ch_rem >= 4) {
                _mm_storeu_ps(kernel_ptr + ch_sep + channels * kx, _mm256_cvtpd_ps(uv_lo));
            }
            else {
                _mm_maskstore_ps(kernel_ptr + ch_sep + channels * kx, TensorShaderAvxBackend::masktable_m128(ch_rem), _mm256_cvtpd_ps(uv_lo));
            }
        }
    }
}

void TensorShaderAvxBackend::Convolution::ChannelwiseKernelProduct1D(unsigned int channels, unsigned int inwidth,
                                                                     unsigned int batch, unsigned int kwidth, unsigned int stride,
                                                                     AvxArray<float>^ inmap, AvxArray<float>^ outmap, AvxArray<float>^ kernel) {

    Util::CheckDuplicateArray(inmap, kernel, outmap);

    unsigned int outwidth = (inwidth - kwidth) / stride + 1;

    Util::CheckLength(channels * inwidth * batch, inmap);
    Util::CheckLength(channels * outwidth * batch, outmap);
    Util::CheckLength(channels * kwidth, kernel);

    float* inmap_ptr = (float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());
    float* kernel_ptr = (float*)(kernel->Ptr.ToPointer());

    kernelproduct_cw1d(channels, 
                       inwidth, outwidth, kwidth, 
                       stride, batch, 
                       inmap_ptr, outmap_ptr, kernel_ptr);
}
