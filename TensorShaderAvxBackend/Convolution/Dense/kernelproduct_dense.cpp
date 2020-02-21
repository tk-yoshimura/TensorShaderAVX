#include "../../TensorShaderAvxBackend.h"

using namespace System;

void kernelproduct_dense(unsigned int inchannels, unsigned int outchannels, unsigned int batch, 
                         const float* __restrict inmap_ptr, const float* __restrict outmap_ptr, float* __restrict kernel_ptr) {
    
    const unsigned int inch_sep = inchannels & ~7u, inch_rem = inchannels - inch_sep;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(inch_rem);
    
    for (unsigned int outch = 0; outch < outchannels; outch++) {
        for (unsigned int inch = 0; inch < inch_sep; inch += 8) {
            __m256d uv_hi = _mm256_setzero_pd(), uv_lo = _mm256_setzero_pd();

            for (unsigned int th = 0; th < batch; th++) {
                __m256 u = _mm256_loadu_ps(inmap_ptr + inch + inchannels * th);
                __m256d v = _mm256_set1_pd(outmap_ptr[outch + outchannels * th]);

                __m256d u_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(u, 1));
                __m256d u_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(u));

                uv_hi = _mm256_fmadd_pd(u_hi, v, uv_hi);
                uv_lo = _mm256_fmadd_pd(u_lo, v, uv_lo);
            }

            _mm_storeu_ps(kernel_ptr + inch + inchannels * outch, _mm256_cvtpd_ps(uv_lo));
            _mm_storeu_ps(kernel_ptr + inch + inchannels * outch + 4, _mm256_cvtpd_ps(uv_hi));
        }

        if (inch_rem > 0) {
            __m256d uv_hi = _mm256_setzero_pd(), uv_lo = _mm256_setzero_pd();

            for (unsigned int th = 0; th < batch; th++) {
                __m256 u = _mm256_maskload_ps(inmap_ptr + inch_sep + inchannels * th, mask);
                __m256d v = _mm256_set1_pd(outmap_ptr[outch + outchannels * th]);

                __m256d u_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(u, 1));
                __m256d u_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(u));

                uv_hi = _mm256_fmadd_pd(u_hi, v, uv_hi);
                uv_lo = _mm256_fmadd_pd(u_lo, v, uv_lo);
            }

            if (inch_rem > 4) {
                _mm_storeu_ps(kernel_ptr + inch_sep + inchannels * outch, _mm256_cvtpd_ps(uv_lo));
                _mm_maskstore_ps(kernel_ptr + inch_sep + inchannels * outch + 4, TensorShaderAvxBackend::masktable_m128(inch_rem - 4), _mm256_cvtpd_ps(uv_hi));
            }
            else if (inch_rem >= 4) {
                _mm_storeu_ps(kernel_ptr + inch_sep + inchannels * outch, _mm256_cvtpd_ps(uv_lo));
            }
            else {
                _mm_maskstore_ps(kernel_ptr + inch_sep + inchannels * outch, TensorShaderAvxBackend::masktable_m128(inch_rem), _mm256_cvtpd_ps(uv_lo));
            }
        }
    }
}

void TensorShaderAvxBackend::Convolution::KernelProductDense(unsigned int inchannels, unsigned int outchannels, unsigned int batch, 
                                                             AvxArray<float>^ inmap, AvxArray<float>^ outmap, AvxArray<float>^ kernel) {

    Util::CheckDuplicateArray(inmap, kernel, outmap);

    Util::CheckLength(inchannels * batch, inmap);
    Util::CheckLength(outchannels * batch, outmap);
    Util::CheckLength(inchannels * outchannels, kernel);

    const float* inmap_ptr = (const float*)(inmap->Ptr.ToPointer());
    const float* outmap_ptr = (const float*)(outmap->Ptr.ToPointer());
    float* kernel_ptr = (float*)(kernel->Ptr.ToPointer());

    kernelproduct_dense(inchannels, outchannels, batch, inmap_ptr, outmap_ptr, kernel_ptr);
}
