#include "../../TensorShaderAvxBackend.h"

using namespace System;

void lerp(unsigned int length, float* srcc_ptr, float* src1_ptr, float* src2_ptr, float* dst_ptr) {
    const unsigned int j = length & ~7u, k = length - j;

    __m256 ones = _mm256_set1_ps(1);

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 xc = _mm256_loadu_ps(srcc_ptr + i);
        __m256 x1 = _mm256_loadu_ps(src1_ptr + i);
        __m256 x2 = _mm256_loadu_ps(src2_ptr + i);

        __m256 y = _mm256_add_ps(_mm256_mul_ps(x1, xc), _mm256_mul_ps(x2, _mm256_sub_ps(ones, xc)));

        _mm256_storeu_ps(dst_ptr + i, y);
    }

    if (k > 0) {
        __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        __m256 xc = _mm256_maskload_ps(srcc_ptr + j, mask);
        __m256 x1 = _mm256_maskload_ps(src1_ptr + j, mask);
        __m256 x2 = _mm256_maskload_ps(src2_ptr + j, mask);

        __m256 y = _mm256_add_ps(_mm256_mul_ps(x1, xc), _mm256_mul_ps(x2, _mm256_sub_ps(ones, xc)));

        _mm256_maskstore_ps(dst_ptr + j, mask, y);
    }
}

void TensorShaderAvxBackend::Elementwise::Lerp(unsigned int index, unsigned int length, cli::array<float>^ srccondition, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst) {

    Util::CheckOutOfRange(index, length, srccondition, src1, src2, dst);

    if (length == 1) {
        dst[index] = src1[0] * srccondition[0] + src2[0] * (1 - srccondition[0]);
        return;
    }

    pin_ptr<float> pinptr_srcc = &srccondition[0];
    pin_ptr<float> pinptr_src1 = &src1[0];
    pin_ptr<float> pinptr_src2 = &src2[0];
    pin_ptr<float> pinptr_dst = &dst[0];

    float* srcc_ptr = pinptr_srcc;
    float* src1_ptr = pinptr_src1;
    float* src2_ptr = pinptr_src2;
    float* dst_ptr = pinptr_dst;

    lerp(length, srcc_ptr + index, src1_ptr + index, src2_ptr + index, dst_ptr + index);
}
