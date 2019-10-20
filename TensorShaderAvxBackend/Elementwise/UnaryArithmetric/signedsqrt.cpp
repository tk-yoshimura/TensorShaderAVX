#include "../../TensorShaderAvxBackend.h"

using namespace System;

void signedsqrt(unsigned int length, float* src_ptr, float* dst_ptr) {
    const unsigned int j = length & ~7u, k = length - j;

    union {
        float f;
        unsigned int i;
    }m32_abs, m32_sign;
    m32_abs.i = 0x7FFFFFFFu;
    m32_sign.i = 0x80000000u;

    __m256 bitmask_abs = _mm256_set1_ps(m32_abs.f);
    __m256 bitmask_sign = _mm256_set1_ps(m32_sign.f);

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x = _mm256_loadu_ps(src_ptr + i);

        __m256 y = _mm256_or_ps(
            _mm256_and_ps(bitmask_sign, x), 
            _mm256_sqrt_ps(_mm256_and_ps(bitmask_abs, x))
        );

        _mm256_storeu_ps(dst_ptr + i, y);
    }

    if (k > 0) {
        __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        __m256 x = _mm256_maskload_ps(src_ptr + j, mask);

        __m256 y = _mm256_or_ps(
            _mm256_and_ps(bitmask_sign, x),
            _mm256_sqrt_ps(_mm256_and_ps(bitmask_abs, x))
        );

        _mm256_maskstore_ps(dst_ptr + j, mask, y);
    }
}

void TensorShaderAvxBackend::Elementwise::SignedSqrt(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst) {
    
    Util::CheckOutOfRange(index, length, src, dst);
    
    if (length == 1) {
        dst[index] = float::IsNaN(src[index]) ? (float)0 : (float)Math::Sign(src[index])
                   * (float)Math::Sqrt(Math::Abs(src[index]));
        return;
    }

    pin_ptr<float> pinptr_src = &src[0];
    pin_ptr<float> pinptr_dst = &dst[0];

    float* src_ptr = pinptr_src;
    float* dst_ptr = pinptr_dst;

    signedsqrt(length, src_ptr + index, dst_ptr + index);
}