#include "../../TensorShaderAvxBackend.h"

using namespace System;

__forceinline __m256 _mm256_signedpow_constant_ps(__m256 x, float c) {
    __m256 fillc = _mm256_set1_ps(c);

    union {
        float f;
        unsigned int i;
    }m32_abs, m32_sign;
    m32_abs.i = 0x7FFFFFFFu;
    m32_sign.i = 0x80000000u;

    __m256 bitmask_abs = _mm256_set1_ps(m32_abs.f);
    __m256 bitmask_sign = _mm256_set1_ps(m32_sign.f);

    __m256 y = _mm256_or_ps(
        _mm256_and_ps(bitmask_sign, x),
        _mm256_pow_ps(_mm256_and_ps(bitmask_abs, x), fillc)
    );

    return y;
}

void signedpow_constant(unsigned int length, float c, const float* __restrict src_ptr, float* __restrict dst_ptr) {
    const unsigned int j = length & ~7u, k = length - j;

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x = _mm256_load_ps(src_ptr + i);

        __m256 y = _mm256_signedpow_constant_ps(x, c);

        _mm256_store_ps(dst_ptr + i, y);
    }

    if (k > 0) {
        __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        __m256 x = _mm256_maskload_ps(src_ptr + j, mask);

        __m256 y = _mm256_signedpow_constant_ps(x, c);

        _mm256_maskstore_ps(dst_ptr + j, mask, y);
    }
}

void TensorShaderAvxBackend::Elementwise::SignedPowConstant(unsigned int length, float c, AvxArray<float>^ src, AvxArray<float>^ dst) {
    
    Util::CheckLength(length, src, dst);
    
    float* src_ptr = (float*)(src->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    signedpow_constant(length, c, src_ptr, dst_ptr);
}
