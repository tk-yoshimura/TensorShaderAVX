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

void max_1(unsigned int src_length, float* src_ptr, unsigned int dst_length, float* dst_ptr, unsigned int slides) {
    const unsigned int j = src_length & ~7u, k = src_length - j;
    const __m256 vmin = _mm256_set1_ps(-HUGE_VALF);
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);
    const __m128i mask1 = TensorShaderAvxBackend::masktable_m128(1);

    while (slides > 0) {
        __m256 u = vmin;

        for (unsigned int i = 0; i < j; i += 8) {
            __m256 x = _mm256_loadu_ps(src_ptr + i);

            u = _mm256_max_ps(u, x);
        }

        if (k > 0) {
            __m256 x = _mm256_maskloadnz_ps(src_ptr + j, vmin, mask);

            u = _mm256_max_ps(u, x);
        }

        __m128 u_hi = _mm256_extractf128_ps(u, 1);
        __m128 u_lo = _mm256_castps256_ps128(u);

        __m128 v = _mm_max_ps(u_hi, u_lo);

        __m128 w = _mm_max_ps(v, _mm_permute_ps(v, _MM_PERM_CDAB));

        __m128 y = _mm_max_ps(w, _mm_permute_ps(w, _MM_PERM_AACC));

        _mm_maskstore_ps(dst_ptr, mask1, y);

        src_ptr += src_length;
        dst_ptr += dst_length;
        slides--;
    }
}

void max_2(unsigned int src_length, float* src_ptr, unsigned int dst_length, float* dst_ptr, unsigned int slides) {
    const unsigned int j = src_length & ~7u, k = src_length - j;
    const __m256 vmin = _mm256_set1_ps(-HUGE_VALF);
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);
    const __m128i mask2 = TensorShaderAvxBackend::masktable_m128(2);

    while (slides > 0) {
        __m256 u = vmin;

        for (unsigned int i = 0; i < j; i += 8) {
            __m256 x = _mm256_loadu_ps(src_ptr + i);

            u = _mm256_max_ps(u, x);
        }

        if (k > 0) {
            __m256 x = _mm256_maskloadnz_ps(src_ptr + j, vmin, mask);

            u = _mm256_max_ps(u, x);
        }

        __m128 u_hi = _mm256_extractf128_ps(u, 1);
        __m128 u_lo = _mm256_castps256_ps128(u);

        __m128 v = _mm_max_ps(u_hi, u_lo);

        __m128 y = _mm_max_ps(v, _mm_permute_ps(v, _MM_PERM_BADC));

        _mm_maskstore_ps(dst_ptr, mask2, y);

        src_ptr += src_length;
        dst_ptr += dst_length;
        slides--;
    }
}

void max_3(unsigned int src_length, float* src_ptr, unsigned int dst_length, float* dst_ptr, unsigned int slides) {
    const __m128 vmin = _mm_set1_ps(-HUGE_VALF);
    const __m128i mask3 = TensorShaderAvxBackend::masktable_m128(3);

    while (slides > 0) {
        __m128 u = vmin;

        for (unsigned int i = 0; i < src_length; i += 3) {
            __m128 x = _mm_maskload_ps(src_ptr + i, mask3);

            u = _mm_max_ps(u, x);
        }

        _mm_maskstore_ps(dst_ptr, mask3, u);

        src_ptr += src_length;
        dst_ptr += dst_length;
        slides--;
    }
}

void max_4(unsigned int src_length, float* src_ptr, unsigned int dst_length, float* dst_ptr, unsigned int slides) {
    const unsigned int j = src_length & ~7u, k = src_length - j;
    const __m256 vmin = _mm256_set1_ps(-HUGE_VALF);
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

    while (slides > 0) {
        __m256 u = vmin;

        for (unsigned int i = 0; i < j; i += 8) {
            __m256 x = _mm256_loadu_ps(src_ptr + i);

            u = _mm256_max_ps(u, x);
        }

        if (k > 0) {
            __m256 x = _mm256_maskloadnz_ps(src_ptr + j, vmin, mask);

            u = _mm256_max_ps(u, x);
        }

        __m128 u_hi = _mm256_extractf128_ps(u, 1);
        __m128 u_lo = _mm256_castps256_ps128(u);

        __m128 v = _mm_max_ps(u_hi, u_lo);

        _mm_storeu_ps(dst_ptr, v);

        src_ptr += src_length;
        dst_ptr += dst_length;
        slides--;
    }
}

void max_5_7(unsigned int src_length, float* src_ptr, unsigned int dst_length, float* dst_ptr, unsigned int slides) {
    const __m256 vmin = _mm256_set1_ps(-HUGE_VALF);
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(dst_length);

    while (slides > 0) {
        __m256 u = vmin;

        for (unsigned int i = 0; i < src_length; i += dst_length) {
            __m256 x = _mm256_maskload_ps(src_ptr + i, mask);

            u = _mm256_max_ps(u, x);
        }

        _mm256_maskstore_ps(dst_ptr, mask, u);

        src_ptr += src_length;
        dst_ptr += dst_length;
        slides--;
    }
}

void max_8(unsigned int src_length, float* src_ptr, unsigned int dst_length, float* dst_ptr, unsigned int slides) {
    const __m256 vmin = _mm256_set1_ps(-HUGE_VALF);

    while (slides > 0) {
        __m256 u = vmin;

        for (unsigned int i = 0; i < src_length; i += dst_length) {
            __m256 x = _mm256_loadu_ps(src_ptr + i);

            u = _mm256_max_ps(u, x);
        }

        _mm256_storeu_ps(dst_ptr, u);

        src_ptr += src_length;
        dst_ptr += dst_length;
        slides--;
    }
}

void max_9_(unsigned int src_length, float* src_ptr, unsigned int dst_length, float* dst_ptr, unsigned int slides) {
    const unsigned int r = dst_length & ~7u, s = dst_length - r;
    const __m256 vmin = _mm256_set1_ps(-HUGE_VALF);
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(s);

    while (slides > 0) {
        for (unsigned int j = 0; j < r; j += 8) {
            __m256 u = vmin;

            for (unsigned int i = 0; i < src_length; i += dst_length) {
                __m256 x = _mm256_loadu_ps(src_ptr + i + j);

                u = _mm256_max_ps(u, x);
            }

            _mm256_storeu_ps(dst_ptr + j, u);
        }
        if (s > 0) {
            __m256 u = vmin;

            for (unsigned int i = 0; i < src_length; i += dst_length) {
                __m256 x = _mm256_maskload_ps(src_ptr + i + r, mask);

                u = _mm256_max_ps(u, x);
            }

            _mm256_maskstore_ps(dst_ptr + r, mask, u);
        }

        src_ptr += src_length;
        dst_ptr += dst_length;
        slides--;
    }
}

void TensorShaderAvxBackend::Aggregation::Max(unsigned int src_length, AvxArray<float>^ src, unsigned int dst_length, AvxArray<float>^ dst, unsigned int slides) {

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

    float* src_ptr = (float*)(src->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    if (dst_length <= 1) {
        max_1(src_length, src_ptr, dst_length, dst_ptr, slides);
    }
    else if (dst_length <= 2) {
        max_2(src_length, src_ptr, dst_length, dst_ptr, slides);
    }
    else if (dst_length <= 3) {
        max_3(src_length, src_ptr, dst_length, dst_ptr, slides);
    }
    else if (dst_length <= 4) {
        max_4(src_length, src_ptr, dst_length, dst_ptr, slides);
    }
    else if (dst_length <= 7) {
        max_5_7(src_length, src_ptr, dst_length, dst_ptr, slides);
    }
    else if (dst_length <= 8) {
        max_8(src_length, src_ptr, dst_length, dst_ptr, slides);
    }
    else {
        max_9_(src_length, src_ptr, dst_length, dst_ptr, slides);
    }
}