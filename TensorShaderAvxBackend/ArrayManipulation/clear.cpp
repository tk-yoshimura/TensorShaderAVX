#include "../TensorShaderAvxBackend.h"

using namespace System;

void clear(unsigned int length, float c, float* __restrict dst_ptr) {
    const unsigned int j = length & ~7u, k = length - j;

    __m256 fillc = _mm256_set1_ps(c);

    for (unsigned int i = 0; i < j; i += 8) {
        _mm256_store_ps(dst_ptr + i, fillc);
    }

    if (k > 0) {
        __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        _mm256_maskstore_ps(dst_ptr + j, mask, fillc);
    }
}

void TensorShaderAvxBackend::ArrayManipulation::Clear(unsigned int length, float c, AvxArray<float>^ dst) {
    
    Util::CheckLength(length, dst);

    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    clear(length, c, dst_ptr);
}