#include "../../TensorShaderAvxBackend.h"

using namespace System;

void logical_not(unsigned int length, const float* __restrict src_ptr, float* __restrict dst_ptr) {
    const unsigned int j = length & ~7u, k = length - j;

    __m256 ones = _mm256_set1_ps(1);

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x = _mm256_load_ps(src_ptr + i);

        __m256 y = _mm256_sub_ps(ones, x);

        _mm256_store_ps(dst_ptr + i, y);
    }

    if (k > 0) {
        __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        __m256 x = _mm256_maskload_ps(src_ptr + j, mask);

        __m256 y = _mm256_sub_ps(ones, x);

        _mm256_maskstore_ps(dst_ptr + j, mask, y);
    }
}

void TensorShaderAvxBackend::Elementwise::LogicalNot(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst) {

    Util::CheckLength(length, src, dst);

    const float* src_ptr = (const float*)(src->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    logical_not(length, src_ptr, dst_ptr);
}