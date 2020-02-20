#include "../../TensorShaderAvxBackend.h"

using namespace System;

__forceinline __m256 _mm256_signedpow_ps(__m256 x1, __m256 x2) {
    union {
        float f;
        unsigned int i;
    }m32_abs, m32_sign;
    m32_abs.i = 0x7FFFFFFFu;
    m32_sign.i = 0x80000000u;

    __m256 bitmask_abs = _mm256_set1_ps(m32_abs.f);
    __m256 bitmask_sign = _mm256_set1_ps(m32_sign.f);

    __m256 y = _mm256_or_ps(
        _mm256_and_ps(bitmask_sign, x1),
        _mm256_pow_ps(_mm256_and_ps(bitmask_abs, x1), x2)
    );

    return y;
}

void signedpow(unsigned int length, float* src1_ptr, float* src2_ptr, float* __restrict dst_ptr) {
    const unsigned int j = length & ~7u, k = length - j;

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x1 = _mm256_load_ps(src1_ptr + i);
        __m256 x2 = _mm256_load_ps(src2_ptr + i);

        __m256 y = _mm256_signedpow_ps(x1, x2);

        _mm256_store_ps(dst_ptr + i, y);
    }

    if (k > 0) {
        __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        __m256 x1 = _mm256_maskload_ps(src1_ptr + j, mask);
        __m256 x2 = _mm256_maskload_ps(src2_ptr + j, mask);

        __m256 y = _mm256_signedpow_ps(x1, x2);

        _mm256_maskstore_ps(dst_ptr + j, mask, y);
    }
}

void TensorShaderAvxBackend::Elementwise::SignedPow(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst) {
    
    Util::CheckLength(length, src1, src2, dst);
    
    float* src1_ptr = (float*)(src1->Ptr.ToPointer());
    float* src2_ptr = (float*)(src2->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    signedpow(length, src1_ptr, src2_ptr, dst_ptr);
}