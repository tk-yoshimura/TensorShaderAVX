#include "../../TensorShaderAvxBackend.h"

using namespace System;

__forceinline __m256 _mm256_greaterthanorequal_ps(__m256 x1, __m256 x2) {
    __m256 ones = _mm256_set1_ps(1);

    __m256 y = _mm256_and_ps(ones, _mm256_cmp_ps(x1, x2, _CMP_GE_OS));

    return y;
}

void greaterthanorequal(unsigned int length, float* src1_ptr, float* src2_ptr, float* dst_ptr) {
    const unsigned int j = length & ~7u, k = length - j;
    
    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x1 = _mm256_load_ps(src1_ptr + i);
        __m256 x2 = _mm256_load_ps(src2_ptr + i);

        __m256 y = _mm256_greaterthanorequal_ps(x1, x2);

        _mm256_store_ps(dst_ptr + i, y);
    }

    if (k > 0) {
        __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        __m256 x1 = _mm256_maskload_ps(src1_ptr + j, mask);
        __m256 x2 = _mm256_maskload_ps(src2_ptr + j, mask);

        __m256 y = _mm256_greaterthanorequal_ps(x1, x2);

        _mm256_maskstore_ps(dst_ptr + j, mask, y);
    }
}

void TensorShaderAvxBackend::Elementwise::GreaterThanOrEqual(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst) {
    
    Util::CheckLength(length, src1, src2, dst);
    
    float* src1_ptr = (float*)(src1->Ptr.ToPointer());
    float* src2_ptr = (float*)(src2->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    greaterthanorequal(length, src1_ptr, src2_ptr, dst_ptr);
}
