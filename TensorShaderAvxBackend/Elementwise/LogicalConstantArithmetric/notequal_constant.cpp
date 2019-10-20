#include "../../TensorShaderAvxBackend.h"

using namespace System;

void notequal_constant(unsigned int length, float c, float* src_ptr, float* dst_ptr) {
    const unsigned int j = length & ~7u, k = length - j;

    __m256 fillc = _mm256_set1_ps(c);
    __m256 ones = _mm256_set1_ps(1);

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x = _mm256_loadu_ps(src_ptr + i);

        __m256 y = _mm256_and_ps(ones, _mm256_cmp_ps(fillc, x, _CMP_NEQ_OS));

        _mm256_storeu_ps(dst_ptr + i, y);
    }

    if (k > 0) {
        __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        __m256 x = _mm256_maskload_ps(src_ptr + j, mask);

        __m256 y = _mm256_and_ps(ones, _mm256_cmp_ps(fillc, x, _CMP_NEQ_OS));

        _mm256_maskstore_ps(dst_ptr + j, mask, y);
    }
}

void TensorShaderAvxBackend::Elementwise::NotEqualConstant(unsigned int index, unsigned int length, float c, cli::array<float>^ src, cli::array<float>^ dst) {
    
    Util::CheckOutOfRange(index, length, src, dst);
    
    if (length == 1) {
        dst[index] = c != src[index] ? (float)1 : (float)0;
        return;
    }

    pin_ptr<float> pinptr_src = &src[0];
    pin_ptr<float> pinptr_dst = &dst[0];

    float* src_ptr = pinptr_src;
    float* dst_ptr = pinptr_dst;

    notequal_constant(length, c, src_ptr + index, dst_ptr + index);
}
