#include "../TensorShaderAvxBackend.h"

using namespace System;

void clear(unsigned int length, float c, float* dst_ptr) {
    const unsigned int j = length & ~7u, k = length - j;

    __m256 fillc = _mm256_set1_ps(c);

    for (unsigned int i = 0; i < j; i += 8) {
        _mm256_storeu_ps(dst_ptr + i, fillc);
    }

    if (k > 0) {
        __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        _mm256_maskstore_ps(dst_ptr + j, mask, fillc);
    }
}

void TensorShaderAvxBackend::ArrayManipulation::Clear(unsigned int index, unsigned int length, float c, cli::array<float>^ dst) {
    
    Util::CheckOutOfRange(index, length, dst);
    
    if (length == 1) {
        dst[index] = c;
        return;
    }

    pin_ptr<float> pinptr_dst = &dst[0];

    float* dst_ptr = pinptr_dst;

    clear(length, c, dst_ptr + index);
}