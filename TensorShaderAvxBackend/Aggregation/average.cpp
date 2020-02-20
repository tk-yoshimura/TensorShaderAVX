#include "../TensorShaderAvxBackend.h"

using namespace System;

void average_1(unsigned int src_length, const float* __restrict src_ptr, 
               unsigned int dst_length, float* __restrict dst_ptr, unsigned int slides) {

    const unsigned int j = src_length & ~7u, k = src_length - j;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);
    const __m128i mask1 = TensorShaderAvxBackend::masktable_m128(1);
    const __m128d inv = _mm_set1_pd((double)dst_length / (double)src_length);

    while (slides > 0) {

        // u_hi : e4, e5, e6, e7   v_lo : e0, e1, e2, e3
        __m256d u_hi = _mm256_setzero_pd(), u_lo = _mm256_setzero_pd();

        for (unsigned int i = 0; i < j; i += 8) {
            __m256 x = _mm256_loadu_ps(src_ptr + i);

            __m256d x_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(x, 1));
            __m256d x_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(x));

            u_hi = _mm256_add_pd(u_hi, x_hi);
            u_lo = _mm256_add_pd(u_lo, x_lo);
        }

        if (k > 0) {
            __m256 x = _mm256_maskload_ps(src_ptr + j, mask);

            __m256d x_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(x, 1));
            __m256d x_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(x));

            u_hi = _mm256_add_pd(u_hi, x_hi);
            u_lo = _mm256_add_pd(u_lo, x_lo);
        }

        // u : e0 + e1, e4 + e5, e2 + e3, e6 + e7  
        __m256d u = _mm256_hadd_pd(u_lo, u_hi);

        // v : e0 + e1 + e4 + e5, 0, e2 + e3 + e6 + e7, 0
        __m256d v = _mm256_hadd_pd(u, _mm256_setzero_pd());

        // w : e0 + e1 + e4 + e5 + e2 + e3 + e6 + e7, 0
        __m128d w = _mm_add_pd(_mm256_extractf128_pd(v, 1), _mm256_castpd256_pd128(v));

        __m128 y = _mm_cvtpd_ps(_mm_mul_pd(w, inv));

        _mm_maskstore_ps(dst_ptr, mask1, y);

        src_ptr += src_length;
        dst_ptr += dst_length;
        slides--;
    }
}

void average_2(unsigned int src_length, const float* __restrict src_ptr, 
               unsigned int dst_length, float* __restrict dst_ptr, unsigned int slides) {

    const unsigned int j = src_length & ~7u, k = src_length - j;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);
    const __m128i mask2 = TensorShaderAvxBackend::masktable_m128(2);
    const __m256d inv = _mm256_set1_pd((double)dst_length / (double)src_length);

    while (slides > 0) {

        // u_hi : e4, e5, e6, e7   v_lo : e0, e1, e2, e3
        __m256d u_hi = _mm256_setzero_pd(), u_lo = _mm256_setzero_pd();

        for (unsigned int i = 0; i < j; i += 8) {
            __m256 x = _mm256_loadu_ps(src_ptr + i);

            __m256d x_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(x, 1));
            __m256d x_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(x));

            u_hi = _mm256_add_pd(u_hi, x_hi);
            u_lo = _mm256_add_pd(u_lo, x_lo);
        }

        if (k > 0) {
            __m256 x = _mm256_maskload_ps(src_ptr + j, mask);

            __m256d x_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(x, 1));
            __m256d x_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(x));

            u_hi = _mm256_add_pd(u_hi, x_hi);
            u_lo = _mm256_add_pd(u_lo, x_lo);
        }

        // u : e0 + e4, e1 + e5, e2 + e6, e3 + e7  
        __m256d u = _mm256_add_pd(u_lo, u_hi);

        // v : e0 + e4, e2 + e6, e1 + e5, e3 + e7
        __m256d v = _mm256_permute4x64_pd(u, _MM_PERM_DBCA);

        // w : e0 + e4 + e2 + e6, 0, e1 + e5 + e3 + e7, 0 
        __m256d w = _mm256_hadd_pd(v, _mm256_setzero_pd());

        __m128 y = _mm_permute_ps(_mm256_cvtpd_ps(_mm256_mul_pd(w, inv)), _MM_PERM_DBCA);

        _mm_maskstore_ps(dst_ptr, mask2, y);

        src_ptr += src_length;
        dst_ptr += dst_length;
        slides--;
    }
}

void average_3(unsigned int src_length, const float* __restrict src_ptr, 
               unsigned int dst_length, float* __restrict dst_ptr, unsigned int slides) {

    const __m128i mask = TensorShaderAvxBackend::masktable_m128(3);
    const __m256d inv = _mm256_set1_pd((double)dst_length / (double)src_length);

    while (slides > 0) {
        // u : e0, e1, e2
        __m256d u = _mm256_setzero_pd();

        for (unsigned int i = 0; i < src_length; i += 3) {
            __m128 x = _mm_maskload_ps(src_ptr + i, mask);

            u = _mm256_add_pd(u, _mm256_cvtps_pd(x));
        }
        __m128 y = _mm256_cvtpd_ps(_mm256_mul_pd(u, inv));

        _mm_maskstore_ps(dst_ptr, mask, y);

        src_ptr += src_length;
        dst_ptr += dst_length;
        slides--;
    }
}

void average_4(unsigned int src_length, const float* __restrict src_ptr, 
               unsigned int dst_length, float* __restrict dst_ptr, unsigned int slides) {

    const unsigned int j = src_length & ~7u, k = src_length - j;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);
    const __m256d inv = _mm256_set1_pd((double)dst_length / (double)src_length);

    while (slides > 0) {

        // u_hi : e4, e5, e6, e7   v_lo : e0, e1, e2, e3
        __m256d u_hi = _mm256_setzero_pd(), u_lo = _mm256_setzero_pd();

        for (unsigned int i = 0; i < j; i += 8) {
            __m256 x = _mm256_loadu_ps(src_ptr + i);

            __m256d x_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(x, 1));
            __m256d x_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(x));

            u_hi = _mm256_add_pd(u_hi, x_hi);
            u_lo = _mm256_add_pd(u_lo, x_lo);
        }

        if (k > 0) {
            __m256 x = _mm256_maskload_ps(src_ptr + j, mask);

            __m256d x_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(x, 1));
            __m256d x_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(x));

            u_hi = _mm256_add_pd(u_hi, x_hi);
            u_lo = _mm256_add_pd(u_lo, x_lo);
        }

        // u : e0 + e4, e1 + e5, e2 + e6, e3 + e7  
        __m256d u = _mm256_add_pd(u_lo, u_hi);

        __m128 y = _mm256_cvtpd_ps(_mm256_mul_pd(u, inv));

        _mm_storeu_ps(dst_ptr, y);

        src_ptr += src_length;
        dst_ptr += dst_length;
        slides--;
    }
}

void average_5_7(unsigned int src_length, const float* __restrict src_ptr, 
                 unsigned int dst_length, float* __restrict dst_ptr, unsigned int slides) {

    const __m256i mask = TensorShaderAvxBackend::masktable_m256(dst_length);
    const __m128i mask_rem = TensorShaderAvxBackend::masktable_m128(dst_length - 4);
    const __m256d inv = _mm256_set1_pd((double)dst_length / (double)src_length);

    while (slides > 0) {
        // u_hi : e4, e5, e6, e7   v_lo : e0, e1, e2, e3
        __m256d u_hi = _mm256_setzero_pd(), u_lo = _mm256_setzero_pd();

        for (unsigned int i = 0; i < src_length; i += dst_length) {
            __m256 x = _mm256_maskload_ps(src_ptr + i, mask);

            __m256d x_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(x, 1));
            __m256d x_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(x));

            u_hi = _mm256_add_pd(u_hi, x_hi);
            u_lo = _mm256_add_pd(u_lo, x_lo);
        }

        __m128 y_hi = _mm256_cvtpd_ps(_mm256_mul_pd(u_hi, inv));
        __m128 y_lo = _mm256_cvtpd_ps(_mm256_mul_pd(u_lo, inv));

        _mm_storeu_ps(dst_ptr, y_lo);
        _mm_maskstore_ps(dst_ptr + 4, mask_rem, y_hi);

        src_ptr += src_length;
        dst_ptr += dst_length;
        slides--;
    }
}

void average_8(unsigned int src_length, const float* __restrict src_ptr, 
               unsigned int dst_length, float* __restrict dst_ptr, unsigned int slides) {

    const __m256d inv = _mm256_set1_pd((double)dst_length / (double)src_length);

    while (slides > 0) {
        // u_hi : e4, e5, e6, e7   v_lo : e0, e1, e2, e3
        __m256d u_hi = _mm256_setzero_pd(), u_lo = _mm256_setzero_pd();

        for (unsigned int i = 0; i < src_length; i += dst_length) {
            __m256 x = _mm256_loadu_ps(src_ptr + i);

            __m256d x_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(x, 1));
            __m256d x_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(x));

            u_hi = _mm256_add_pd(u_hi, x_hi);
            u_lo = _mm256_add_pd(u_lo, x_lo);
        }

        __m128 y_hi = _mm256_cvtpd_ps(_mm256_mul_pd(u_hi, inv));
        __m128 y_lo = _mm256_cvtpd_ps(_mm256_mul_pd(u_lo, inv));

        _mm_storeu_ps(dst_ptr, y_lo);
        _mm_storeu_ps(dst_ptr + 4, y_hi);

        src_ptr += src_length;
        dst_ptr += dst_length;
        slides--;
    }
}

void average_9_(unsigned int src_length, const float* __restrict src_ptr, 
                unsigned int dst_length, float* __restrict dst_ptr, unsigned int slides) {

    const unsigned int r = dst_length & ~7u, s = dst_length - r;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(s);
    const __m128i masks = TensorShaderAvxBackend::masktable_m128(s & 3);
    const __m256d inv = _mm256_set1_pd((double)dst_length / (double)src_length);

    while (slides > 0) {

        for (unsigned int j = 0; j < r; j += 8) {
            // u_hi : e4, e5, e6, e7   v_lo : e0, e1, e2, e3
            __m256d u_hi = _mm256_setzero_pd(), u_lo = _mm256_setzero_pd();

            for (unsigned int i = 0; i < src_length; i += dst_length) {
                __m256 x = _mm256_loadu_ps(src_ptr + i + j);

                __m256d x_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(x, 1));
                __m256d x_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(x));

                u_hi = _mm256_add_pd(u_hi, x_hi);
                u_lo = _mm256_add_pd(u_lo, x_lo);
            }

            __m128 y_hi = _mm256_cvtpd_ps(_mm256_mul_pd(u_hi, inv));
            __m128 y_lo = _mm256_cvtpd_ps(_mm256_mul_pd(u_lo, inv));

            _mm_storeu_ps(dst_ptr + j, y_lo);
            _mm_storeu_ps(dst_ptr + j + 4, y_hi);
        }
        if (s > 0) {
            // u_hi : e4, e5, e6, e7   v_lo : e0, e1, e2, e3
            __m256d u_hi = _mm256_setzero_pd(), u_lo = _mm256_setzero_pd();

            for (unsigned int i = 0; i < src_length; i += dst_length) {
                __m256 x = _mm256_maskload_ps(src_ptr + i + r, mask);

                __m256d x_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(x, 1));
                __m256d x_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(x));

                u_hi = _mm256_add_pd(u_hi, x_hi);
                u_lo = _mm256_add_pd(u_lo, x_lo);
            }

            __m128 y_hi = _mm256_cvtpd_ps(_mm256_mul_pd(u_hi, inv));
            __m128 y_lo = _mm256_cvtpd_ps(_mm256_mul_pd(u_lo, inv));

            if (s > 4) {
                _mm_storeu_ps(dst_ptr + r, y_lo);
                _mm_maskstore_ps(dst_ptr + r + 4, masks, y_hi);
            }
            else if (s >= 4) {
                _mm_storeu_ps(dst_ptr + r, y_lo);
            }
            else {
                _mm_maskstore_ps(dst_ptr + r, masks, y_lo);
            }
        }

        src_ptr += src_length;
        dst_ptr += dst_length;
        slides--;
    }
}

void TensorShaderAvxBackend::Aggregation::Average(unsigned int src_length, AvxArray<float>^ src, unsigned int dst_length, AvxArray<float>^ dst, unsigned int slides) {

    Util::CheckDuplicateArray(src, dst);

    if (src_length == 0 && dst_length == 0) {
        return;
    }

    if (slides == 0) {
        return;
    }

    if (((src_length * slides) / slides) != src_length) {
        throw gcnew System::OverflowException();
    }

    if (((dst_length * slides) / slides) != dst_length) {
        throw gcnew System::OverflowException();
    }

    Util::CheckLength(src_length * slides, src);
    Util::CheckLength(dst_length * slides, dst);

    if (src_length < 1 || dst_length < 1 || src_length % dst_length != 0) {
        throw gcnew System::ArgumentException();
    }

    const float* src_ptr = (const float*)(src->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    if (dst_length <= 1) {
        average_1(src_length, src_ptr, dst_length, dst_ptr, slides);
    }
    else if (dst_length <= 2) {
        average_2(src_length, src_ptr, dst_length, dst_ptr, slides);
    }
    else if (dst_length <= 3) {
        average_3(src_length, src_ptr, dst_length, dst_ptr, slides);
    }
    else if (dst_length <= 4) {
        average_4(src_length, src_ptr, dst_length, dst_ptr, slides);
    }
    else if (dst_length <= 7) {
        average_5_7(src_length, src_ptr, dst_length, dst_ptr, slides);
    }
    else if (dst_length <= 8) {
        average_8(src_length, src_ptr, dst_length, dst_ptr, slides);
    }
    else {
        average_9_(src_length, src_ptr, dst_length, dst_ptr, slides);
    }
}