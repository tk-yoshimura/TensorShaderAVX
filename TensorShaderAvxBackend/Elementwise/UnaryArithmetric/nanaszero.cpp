#include "../../TensorShaderAvxBackend.h"

using namespace System;

__forceinline __m256 _mm256_nanaszero_ps(__m256 x) {
    __m256 y = _mm256_and_ps(x, _mm256_cmp_ps(x, x, _CMP_EQ_OS));

    return y;
}

void nanaszero(unsigned int length, float* src_ptr, float* dst_ptr) {
    const unsigned int j = length & ~7u, k = length - j;

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x = _mm256_load_ps(src_ptr + i);

        __m256 y = _mm256_nanaszero_ps(x);

        _mm256_store_ps(dst_ptr + i, y);
    }

    if (k > 0) {
        __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        __m256 x = _mm256_maskload_ps(src_ptr + j, mask);

        __m256 y = _mm256_nanaszero_ps(x);

        _mm256_maskstore_ps(dst_ptr + j, mask, y);
    }
}

void TensorShaderAvxBackend::Elementwise::NanAsZero(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst) {
    
    Util::CheckLength(length, src, dst);
    
    float* src_ptr = (float*)(src->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    nanaszero(length, src_ptr, dst_ptr);
}