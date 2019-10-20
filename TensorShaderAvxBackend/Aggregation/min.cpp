#include "../TensorShaderAvxBackend.h"

#include <limits>

using namespace System;

__forceinline __m256 _mm256_maskloadnz_ps(float* ptr, __m256 nz, __m256i mask) {
    union {
        float f;
        unsigned int i;
    }m32;
    m32.i = 0xFFFFFFFFu;

    __m256 not = _mm256_set1_ps(m32.f);

    return _mm256_or_ps(
        _mm256_maskload_ps(ptr, mask),
        _mm256_and_ps(nz, _mm256_xor_ps(not, _mm256_castsi256_ps(mask)))
    );
}

void min_1(unsigned int src_length, float* src_ptr, unsigned int dst_length, float* dst_ptr, unsigned int slides) {
    const unsigned int j = src_length & ~7u, k = src_length - j;
    const __m256 vmax = _mm256_set1_ps(HUGE_VALF);
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);
    const __m128i mask1 = TensorShaderAvxBackend::masktable_m128(1);

    while (slides > 0) {
        __m256 u = vmax;

        for (unsigned int i = 0; i < j; i += 8) {
            __m256 x = _mm256_loadu_ps(src_ptr + i);

            u = _mm256_min_ps(u, x);
        }

        if (k > 0) {
            __m256 x = _mm256_maskloadnz_ps(src_ptr + j, vmax, mask);

            u = _mm256_min_ps(u, x);
        }

        __m128 u_hi = _mm256_extractf128_ps(u, 1);
        __m128 u_lo = _mm256_castps256_ps128(u);

        __m128 v = _mm_min_ps(u_hi, u_lo);

        __m128 w = _mm_min_ps(v, _mm_permute_ps(v, _MM_PERM_CDAB));

        __m128 y = _mm_min_ps(w, _mm_permute_ps(w, _MM_PERM_AACC));

        _mm_maskstore_ps(dst_ptr, mask1, y);

        src_ptr += src_length;
        dst_ptr += dst_length;
        slides--;
    }
}

void min_2(unsigned int src_length, float* src_ptr, unsigned int dst_length, float* dst_ptr, unsigned int slides) {
    const unsigned int j = src_length & ~7u, k = src_length - j;
    const __m256 vmax = _mm256_set1_ps(HUGE_VALF);
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);
    const __m128i mask2 = TensorShaderAvxBackend::masktable_m128(2);

    while (slides > 0) {
        __m256 u = vmax;

        for (unsigned int i = 0; i < j; i += 8) {
            __m256 x = _mm256_loadu_ps(src_ptr + i);

            u = _mm256_min_ps(u, x);
        }

        if (k > 0) {
            __m256 x = _mm256_maskloadnz_ps(src_ptr + j, vmax, mask);

            u = _mm256_min_ps(u, x);
        }

        __m128 u_hi = _mm256_extractf128_ps(u, 1);
        __m128 u_lo = _mm256_castps256_ps128(u);

        __m128 v = _mm_min_ps(u_hi, u_lo);

        __m128 y = _mm_min_ps(v, _mm_permute_ps(v, _MM_PERM_BADC));

        _mm_maskstore_ps(dst_ptr, mask2, y);

        src_ptr += src_length;
        dst_ptr += dst_length;
        slides--;
    }
}

void min_3(unsigned int src_length, float* src_ptr, unsigned int dst_length, float* dst_ptr, unsigned int slides) {
    const __m128 vmax = _mm_set1_ps(HUGE_VALF);
    const __m128i mask3 = TensorShaderAvxBackend::masktable_m128(3);

    while (slides > 0) {
        __m128 u = vmax;

        for (unsigned int i = 0; i < src_length; i += 3) {
            __m128 x = _mm_maskload_ps(src_ptr + i, mask3);

            u = _mm_min_ps(u, x);
        }

        _mm_maskstore_ps(dst_ptr, mask3, u);

        src_ptr += src_length;
        dst_ptr += dst_length;
        slides--;
    }
}

void min_4(unsigned int src_length, float* src_ptr, unsigned int dst_length, float* dst_ptr, unsigned int slides) {
    const unsigned int j = src_length & ~7u, k = src_length - j;
    const __m256 vmax = _mm256_set1_ps(HUGE_VALF);
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

    while (slides > 0) {
        __m256 u = vmax;

        for (unsigned int i = 0; i < j; i += 8) {
            __m256 x = _mm256_loadu_ps(src_ptr + i);

            u = _mm256_min_ps(u, x);
        }

        if (k > 0) {
            __m256 x = _mm256_maskloadnz_ps(src_ptr + j, vmax, mask);

            u = _mm256_min_ps(u, x);
        }

        __m128 u_hi = _mm256_extractf128_ps(u, 1);
        __m128 u_lo = _mm256_castps256_ps128(u);

        __m128 v = _mm_min_ps(u_hi, u_lo);

        _mm_storeu_ps(dst_ptr, v);

        src_ptr += src_length;
        dst_ptr += dst_length;
        slides--;
    }
}

void min_5_7(unsigned int src_length, float* src_ptr, unsigned int dst_length, float* dst_ptr, unsigned int slides) {
    const __m256 vmax = _mm256_set1_ps(HUGE_VALF);
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(dst_length);

    while (slides > 0) {
        __m256 u = vmax;

        for (unsigned int i = 0; i < src_length; i += dst_length) {
            __m256 x = _mm256_maskload_ps(src_ptr + i, mask);

            u = _mm256_min_ps(u, x);
        }

        _mm256_maskstore_ps(dst_ptr, mask, u);

        src_ptr += src_length;
        dst_ptr += dst_length;
        slides--;
    }
}

void min_8(unsigned int src_length, float* src_ptr, unsigned int dst_length, float* dst_ptr, unsigned int slides) {
    const __m256 vmax = _mm256_set1_ps(HUGE_VALF);

    while (slides > 0) {
        __m256 u = vmax;

        for (unsigned int i = 0; i < src_length; i += dst_length) {
            __m256 x = _mm256_loadu_ps(src_ptr + i);

            u = _mm256_min_ps(u, x);
        }

        _mm256_storeu_ps(dst_ptr, u);

        src_ptr += src_length;
        dst_ptr += dst_length;
        slides--;
    }
}

void min_9_(unsigned int src_length, float* src_ptr, unsigned int dst_length, float* dst_ptr, unsigned int slides) {
    const unsigned int r = dst_length & ~7u, s = dst_length - r;
    const __m256 vmax = _mm256_set1_ps(HUGE_VALF);
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(s);

    while (slides > 0) {
        for (unsigned int j = 0; j < r; j += 8) {
            __m256 u = vmax;

            for (unsigned int i = 0; i < src_length; i += dst_length) {
                __m256 x = _mm256_loadu_ps(src_ptr + i + j);

                u = _mm256_min_ps(u, x);
            }

            _mm256_storeu_ps(dst_ptr + j, u);
        }
        if (s > 0) {
            __m256 u = vmax;

            for (unsigned int i = 0; i < src_length; i += dst_length) {
                __m256 x = _mm256_maskload_ps(src_ptr + i + r, mask);

                u = _mm256_min_ps(u, x);
            }

            _mm256_maskstore_ps(dst_ptr + r, mask, u);
        }

        src_ptr += src_length;
        dst_ptr += dst_length;
        slides--;
    }
}

void TensorShaderAvxBackend::Aggregation::Min(unsigned int src_index, unsigned int src_length, cli::array<float>^ src, unsigned int dst_index, unsigned int dst_length, cli::array<float>^ dst, unsigned int slides) {

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

    Util::CheckOutOfRange(src_index, src_length * slides, src);
    Util::CheckOutOfRange(dst_index, dst_length * slides, dst);

    if (src_length < 1 || dst_length < 1 || src_length % dst_length != 0) {
        throw gcnew System::ArgumentException();
    }

    pin_ptr<float> pinptr_src = &src[0];
    pin_ptr<float> pinptr_dst = &dst[0];

    float* src_ptr = pinptr_src;
    float* dst_ptr = pinptr_dst;

    if (dst_length <= 1) {
        min_1(src_length, src_ptr + src_index, dst_length, dst_ptr + dst_index, slides);
    }
    else if (dst_length <= 2) {
        min_2(src_length, src_ptr + src_index, dst_length, dst_ptr + dst_index, slides);
    }
    else if (dst_length <= 3) {
        min_3(src_length, src_ptr + src_index, dst_length, dst_ptr + dst_index, slides);
    }
    else if (dst_length <= 4) {
        min_4(src_length, src_ptr + src_index, dst_length, dst_ptr + dst_index, slides);
    }
    else if (dst_length <= 7) {
        min_5_7(src_length, src_ptr + src_index, dst_length, dst_ptr + dst_index, slides);
    }
    else if (dst_length <= 8) {
        min_8(src_length, src_ptr + src_index, dst_length, dst_ptr + dst_index, slides);
    }
    else {
        min_9_(src_length, src_ptr + src_index, dst_length, dst_ptr + dst_index, slides);
    }
}