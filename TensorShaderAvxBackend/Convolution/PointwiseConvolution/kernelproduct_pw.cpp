#include "../../TensorShaderAvxBackend.h"

using namespace System;

void kernelproduct_pw(unsigned int inchannels, unsigned int outchannels,
                      unsigned points, unsigned batch, unsigned int outch,
                      float* inmap_ptr, float* outmap_ptr, float* kernel_ptr) {

    const unsigned int inch_sep = inchannels & ~7u, inch_rem = inchannels - inch_sep;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(inch_rem);
    
    for (unsigned int inch = 0; inch < inch_sep; inch += 8) {

        __m256d uv_hi = _mm256_setzero_pd(), uv_lo = _mm256_setzero_pd();

        for (unsigned int th = 0; th < batch; th++) {
            for (unsigned int i = 0; i < points; i++) {

                __m256 u = _mm256_loadu_ps(inmap_ptr + inch + inchannels * (i + points * th));
                __m256d v = _mm256_set1_pd(outmap_ptr[outch + outchannels * (i + points * th)]);

                __m256d u_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(u, 1));
                __m256d u_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(u));

                uv_hi = _mm256_fmadd_pd(u_hi, v, uv_hi);
                uv_lo = _mm256_fmadd_pd(u_lo, v, uv_lo);
            }
        }

        _mm_storeu_ps(kernel_ptr + inch + inchannels * outch, _mm256_cvtpd_ps(uv_lo));
        _mm_storeu_ps(kernel_ptr + inch + inchannels * outch + 4, _mm256_cvtpd_ps(uv_hi));
    }

    if (inch_rem > 0) {

        __m256d uv_hi = _mm256_setzero_pd(), uv_lo = _mm256_setzero_pd();

        for (unsigned int th = 0; th < batch; th++) {
            for (unsigned int i = 0; i < points; i++) {
                __m256 u = _mm256_maskload_ps(inmap_ptr + inch_sep + inchannels * (i + points * th), mask);
                __m256d v = _mm256_set1_pd(outmap_ptr[outch + outchannels * (i + points * th)]);

                __m256d u_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(u, 1));
                __m256d u_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(u));

                uv_hi = _mm256_fmadd_pd(u_hi, v, uv_hi);
                uv_lo = _mm256_fmadd_pd(u_lo, v, uv_lo);
            }
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

void TensorShaderAvxBackend::Convolution::PointwiseKernelProduct(unsigned int inchannels, unsigned int outchannels, unsigned int points,
                                                                 unsigned int batch, unsigned int outch,
                                                                 cli::array<float>^ inmap, cli::array<float>^ outmap, cli::array<float>^ kernel) {

    Util::CheckDuplicateArray(inmap, kernel, outmap);

    if (inchannels * points * batch > (unsigned int)inmap->Length) {
        throw gcnew System::ArgumentException();
    }
    if (outchannels * points * batch > (unsigned int)outmap->Length) {
        throw gcnew System::ArgumentException();
    }
    if (inchannels * outchannels > (unsigned int)kernel->Length) {
        throw gcnew System::ArgumentException();
    }
    if (outch >= outchannels) {
        throw gcnew System::ArgumentException();
    }

    pin_ptr<float> pinptr_inmap = &inmap[0];
    pin_ptr<float> pinptr_outmap = &outmap[0];
    pin_ptr<float> pinptr_kernel = &kernel[0];

    float* inmap_ptr = pinptr_inmap;
    float* outmap_ptr = pinptr_outmap;
    float* kernel_ptr = pinptr_kernel;

    kernelproduct_pw(inchannels, outchannels, 
                     points, batch, outch,
                     inmap_ptr, outmap_ptr, kernel_ptr);
}
