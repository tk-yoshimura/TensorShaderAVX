#include "../../TensorShaderAvxBackend.h"

using namespace System;

__forceinline __m256 _mm256_abs_ps(__m256 x) {
    union {
        float f;
        unsigned int i;
    }m32;
    m32.i = 0x7FFFFFFFu;

    __m256 bitmask = _mm256_set1_ps(m32.f);

    __m256 y = _mm256_and_ps(bitmask, x);

    return y;
}

void abs(unsigned int length, float* src_ptr, float* dst_ptr) {
    const unsigned int j = length & ~7u, k = length - j;

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x = _mm256_load_ps(src_ptr + i);

        __m256 y = _mm256_abs_ps(x);

        _mm256_store_ps(dst_ptr + i, y);
    }

    if (k > 0) {
        __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        __m256 x = _mm256_maskload_ps(src_ptr + j, mask);

        __m256 y = _mm256_abs_ps(x);

        _mm256_maskstore_ps(dst_ptr + j, mask, y);
    }
}

void TensorShaderAvxBackend::Elementwise::Abs(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst) {
    
    Util::CheckLength(length, src, dst);

    float* src_ptr = (float*)(src->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    abs(length, src_ptr, dst_ptr);
}