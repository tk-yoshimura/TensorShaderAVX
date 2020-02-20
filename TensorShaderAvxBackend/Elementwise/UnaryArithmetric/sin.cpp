#include "../../TensorShaderAvxBackend.h"

using namespace System;

void sin(unsigned int length, const float* __restrict src_ptr, float* __restrict dst_ptr) {
    const unsigned int j = length & ~7u, k = length - j;

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x = _mm256_load_ps(src_ptr + i);

        __m256 y = _mm256_sin_ps(x);

        _mm256_store_ps(dst_ptr + i, y);
    }

    if (k > 0) {
        __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        __m256 x = _mm256_maskload_ps(src_ptr + j, mask);

        __m256 y = _mm256_sin_ps(x);

        _mm256_maskstore_ps(dst_ptr + j, mask, y);
    }
}

void TensorShaderAvxBackend::Elementwise::Sin(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst) {
    
    Util::CheckLength(length, src, dst);

    float* src_ptr = (float*)(src->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    sin(length, src_ptr, dst_ptr);
}