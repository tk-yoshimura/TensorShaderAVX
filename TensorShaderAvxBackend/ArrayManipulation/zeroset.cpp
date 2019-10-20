#include "../TensorShaderAvxBackend.h"

using namespace System;

void zeroset(unsigned int length, float* dst_ptr) {
    const unsigned int j = length & ~7u, k = length - j;

    __m256 zeros = _mm256_setzero_ps();

    for (unsigned int i = 0; i < j; i += 8) {
        _mm256_storeu_ps(dst_ptr + i, zeros);
    }

    if (k > 0) {
        __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        _mm256_maskstore_ps(dst_ptr + j, mask, zeros);
    }
}

void TensorShaderAvxBackend::ArrayManipulation::Zeroset(unsigned int index, unsigned int length, cli::array<float>^ dst) {
    
    Util::CheckOutOfRange(index, length, dst);
    
    if (length == 1) {
        dst[index] = (float)0;
        return;
    }

    pin_ptr<float> pinptr_dst = &dst[0];

    float* dst_ptr = pinptr_dst;

    zeroset(length, dst_ptr + index);
}