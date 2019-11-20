#include "../../TensorShaderAvxBackend.h"

using namespace System;

void convolution_cw1d(unsigned int channels, 
                      unsigned int inwidth, unsigned int outwidth, unsigned int kwidth,
                      unsigned stride, unsigned int th, 
                      float* inmap_ptr, float* outmap_ptr, float* kernel_ptr) {
        
    const unsigned int inmap_offset = channels * inwidth * th, outmap_offset = channels * outwidth * th;
    const unsigned int ch_sep = channels & ~7u, ch_rem = channels - ch_sep;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(ch_rem);
    
    inmap_ptr += inmap_offset;
    outmap_ptr += outmap_offset;

    for (unsigned int ox = 0; ox < outwidth; ox++) {
        for (unsigned int ch = 0; ch < ch_sep; ch += 8) {
            __m256d uv_hi = _mm256_setzero_pd(), uv_lo = _mm256_setzero_pd();

            for (unsigned int kx = 0, ix = ox * stride; kx < kwidth; kx++, ix++) {
                __m256 u = _mm256_loadu_ps(inmap_ptr + ch + channels * ix);
                __m256 v = _mm256_loadu_ps(kernel_ptr + ch + channels * kx);

                __m256d u_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(u, 1));
                __m256d u_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(u));

                __m256d v_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1));
                __m256d v_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(v));

                uv_hi = _mm256_fmadd_pd(u_hi, v_hi, uv_hi);
                uv_lo = _mm256_fmadd_pd(u_lo, v_lo, uv_lo);
            }

            _mm_storeu_ps(outmap_ptr + ch + channels * ox, _mm256_cvtpd_ps(uv_lo));
            _mm_storeu_ps(outmap_ptr + ch + channels * ox + 4, _mm256_cvtpd_ps(uv_hi));
        }

        if (ch_rem > 0) {
            __m256d uv_hi = _mm256_setzero_pd(), uv_lo = _mm256_setzero_pd();

            for (unsigned int kx = 0, ix = ox * stride; kx < kwidth; kx++, ix++) {
                __m256 u = _mm256_maskload_ps(inmap_ptr + ch_sep + channels * ix, mask);
                __m256 v = _mm256_maskload_ps(kernel_ptr + ch_sep + channels * kx, mask);

                __m256d u_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(u, 1));
                __m256d u_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(u));

                __m256d v_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1));
                __m256d v_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(v));

                uv_hi = _mm256_fmadd_pd(u_hi, v_hi, uv_hi);
                uv_lo = _mm256_fmadd_pd(u_lo, v_lo, uv_lo);
            }

            if (ch_rem > 4) {
                _mm_storeu_ps(outmap_ptr + ch_sep + channels * ox, _mm256_cvtpd_ps(uv_lo));
                _mm_maskstore_ps(outmap_ptr + ch_sep + channels * ox + 4, TensorShaderAvxBackend::masktable_m128(ch_rem - 4), _mm256_cvtpd_ps(uv_hi));
            }
            else if (ch_rem >= 4) {
                _mm_storeu_ps(outmap_ptr + ch_sep + channels * ox, _mm256_cvtpd_ps(uv_lo));
            }
            else {
                _mm_maskstore_ps(outmap_ptr + ch_sep + channels * ox, TensorShaderAvxBackend::masktable_m128(ch_rem), _mm256_cvtpd_ps(uv_lo));
            }
        }
    }
}

void TensorShaderAvxBackend::Convolution::ChannelwiseConvolution1D(unsigned int channels, unsigned int inwidth,
                                                                   unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int stride,
                                                                   AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, kernel, outmap);

    if (th >= batch) {
        throw gcnew System::ArgumentException();
    }

    unsigned int outwidth = (inwidth - kwidth) / stride + 1;

    Util::CheckLength(channels * inwidth * batch, inmap);
    Util::CheckLength(channels * outwidth * batch, outmap);
    Util::CheckLength(channels * kwidth, kernel);
    
    float* inmap_ptr = (float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());
    float* kernel_ptr = (float*)(kernel->Ptr.ToPointer());

    convolution_cw1d(channels, 
                     inwidth, outwidth, kwidth, 
                     stride, th, 
                     inmap_ptr, outmap_ptr, kernel_ptr);
}
