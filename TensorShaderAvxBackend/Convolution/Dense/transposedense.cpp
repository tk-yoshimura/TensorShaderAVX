#include "../../TensorShaderAvxBackend.h"

using namespace System;

__forceinline __m128 _mm256d_sum(__m256d hi, __m256d lo) {
    __m256d u = _mm256_hadd_pd(lo, hi);
    __m256d v = _mm256_hadd_pd(u, _mm256_setzero_pd());
    __m128d w = _mm_add_pd(_mm256_extractf128_pd(v, 1), _mm256_castpd256_pd128(v));

    return _mm_cvtpd_ps(w);
}

void transposedense(unsigned int inchannels, unsigned int outchannels, unsigned int th, float* inmap_ptr, float* outmap_ptr, float* kernel_ptr) {
    const unsigned int inmap_offset = inchannels * th, outmap_offset = outchannels * th;
    const unsigned int inch_sep = inchannels & ~7u, inch_rem = inchannels - inch_sep;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(inch_rem);
    const __m128i mask1 = TensorShaderAvxBackend::masktable_m128(1);
    const __m256i index = _mm256_setr_epi32(0, outchannels, outchannels * 2, outchannels * 3, 
                                            outchannels * 4, outchannels * 5, outchannels * 6, outchannels * 7);

    inmap_ptr += inmap_offset;
    outmap_ptr += outmap_offset;

    for (unsigned int outch = 0; outch < outchannels; outch++) {
        __m256d uv_hi = _mm256_setzero_pd(), uv_lo = _mm256_setzero_pd();

        for (unsigned int inch = 0; inch < inch_sep; inch += 8) {
            __m256 u = _mm256_loadu_ps(inmap_ptr + inch);
            __m256 v = _mm256_i32gather_ps(kernel_ptr + outch + outchannels * inch, index, 4);

            __m256d u_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(u, 1));
            __m256d u_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(u));

            __m256d v_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1));
            __m256d v_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(v));

            uv_hi = _mm256_fmadd_pd(u_hi, v_hi, uv_hi);
            uv_lo = _mm256_fmadd_pd(u_lo, v_lo, uv_lo);
        }

        if (inch_rem > 0) {
            __m256 u = _mm256_maskload_ps(inmap_ptr + inch_sep, mask);
            __m256 v = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), kernel_ptr + outch + outchannels * inch_sep,
                                                index, _mm256_castsi256_ps(mask), 4);

            __m256d u_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(u, 1));
            __m256d u_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(u));

            __m256d v_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1));
            __m256d v_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(v));

            uv_hi = _mm256_fmadd_pd(u_hi, v_hi, uv_hi);
            uv_lo = _mm256_fmadd_pd(u_lo, v_lo, uv_lo);
        }

        _mm_maskstore_ps(outmap_ptr + outch, mask1, _mm256d_sum(uv_hi, uv_lo));
    }
}

void TensorShaderAvxBackend::Convolution::TransposeDense(unsigned int inchannels, unsigned int outchannels, unsigned int batch, unsigned int th, 
                                                         AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, kernel, outmap);

    if (th >= batch) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckLength(inchannels * batch, inmap);
    Util::CheckLength(outchannels * batch, outmap);
    Util::CheckLength(inchannels * outchannels, kernel);

    float* inmap_ptr = (float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());
    float* kernel_ptr = (float*)(kernel->Ptr.ToPointer());

    transposedense(inchannels, outchannels, th, inmap_ptr, outmap_ptr, kernel_ptr);
}
