#include "../../TensorShaderAvxBackend.h"

using namespace System;

__forceinline __m256 _mm256_lessthanorequal_constant_ps(float c, __m256 x) {
    __m256 fillc = _mm256_set1_ps(c);
    __m256 ones = _mm256_set1_ps(1);

    __m256 y = _mm256_and_ps(ones, _mm256_cmp_ps(fillc, x, _CMP_LE_OS));

    return y;
}

void lessthanorequal_constant(unsigned int length, float c, float* src_ptr, float* dst_ptr) {
    const unsigned int j = length & ~7u, k = length - j;

    __m256 fillc = _mm256_set1_ps(c);
    __m256 ones = _mm256_set1_ps(1);

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x = _mm256_load_ps(src_ptr + i);

        __m256 y = _mm256_lessthanorequal_constant_ps(c, x);

        _mm256_store_ps(dst_ptr + i, y);
    }

    if (k > 0) {
        __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        __m256 x = _mm256_maskload_ps(src_ptr + j, mask);

        __m256 y = _mm256_lessthanorequal_constant_ps(c, x);

        _mm256_maskstore_ps(dst_ptr + j, mask, y);
    }
}

void TensorShaderAvxBackend::Elementwise::LessThanOrEqualConstant(unsigned int length, float c, AvxArray<float>^ src, AvxArray<float>^ dst) {
    
    Util::CheckLength(length, src, dst);

    float* src_ptr = (float*)(src->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    lessthanorequal_constant(length, c, src_ptr, dst_ptr);
}
