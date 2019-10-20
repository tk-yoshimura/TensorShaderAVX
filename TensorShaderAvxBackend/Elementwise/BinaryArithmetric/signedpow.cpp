#include "../../TensorShaderAvxBackend.h"

using namespace System;

void signedpow(unsigned int length, float* src1_ptr, float* src2_ptr, float* dst_ptr) {
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
        __m256 x1 = _mm256_loadu_ps(src1_ptr + i);
        __m256 x2 = _mm256_loadu_ps(src2_ptr + i);

        __m256 y = _mm256_or_ps(
            _mm256_and_ps(bitmask_sign, x1),
            _mm256_pow_ps(_mm256_and_ps(bitmask_abs, x1), x2)
        );

        _mm256_storeu_ps(dst_ptr + i, y);
    }

    if (k > 0) {
        __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        __m256 x1 = _mm256_maskload_ps(src1_ptr + j, mask);
        __m256 x2 = _mm256_maskload_ps(src2_ptr + j, mask);

        __m256 y = _mm256_or_ps(
            _mm256_and_ps(bitmask_sign, x1),
            _mm256_pow_ps(_mm256_and_ps(bitmask_abs, x1), x2)
        );

        _mm256_maskstore_ps(dst_ptr + j, mask, y);
    }
}

void TensorShaderAvxBackend::Elementwise::SignedPow(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst) {
    
    Util::CheckOutOfRange(index, length, src1, src2, dst);
    
    if (length == 1) {
        dst[index] = (float::IsNaN(src1[index]) ? (float)0 : (float)Math::Sign(src1[index]))
                   * (float)Math::Pow(Math::Abs(src1[index]), src2[index]);
        return;
    }

    pin_ptr<float> pinptr_src1 = &src1[0];
    pin_ptr<float> pinptr_src2 = &src2[0];
    pin_ptr<float> pinptr_dst = &dst[0];

    float* src1_ptr = pinptr_src1;
    float* src2_ptr = pinptr_src2;
    float* dst_ptr = pinptr_dst;

    signedpow(length, src1_ptr + index, src2_ptr + index, dst_ptr + index);
}