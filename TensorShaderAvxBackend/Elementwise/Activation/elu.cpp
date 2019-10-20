#include "../../TensorShaderAvxBackend.h"

using namespace System;

void elu(unsigned int length, float slope, float* src_ptr, float* dst_ptr) {
    const unsigned int j = length & ~7u, k = length - j;

    __m256 zeros = _mm256_setzero_ps();
    __m256 ones = _mm256_set1_ps(1);
    __m256 fills = _mm256_set1_ps(slope);

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x = _mm256_loadu_ps(src_ptr + i);

        __m256 y = _mm256_or_ps(
            _mm256_and_ps(x, _mm256_cmp_ps(x, zeros, _CMP_NLE_UQ)),
            _mm256_and_ps(
                _mm256_mul_ps(fills, _mm256_sub_ps(_mm256_exp_ps(x), ones)),
                _mm256_cmp_ps(x, zeros, _CMP_LT_OS)
            )
        );

        _mm256_storeu_ps(dst_ptr + i, y);
    }

    if (k > 0) {
        __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        __m256 x = _mm256_maskload_ps(src_ptr + j, mask);

        __m256 y = _mm256_or_ps(
            _mm256_and_ps(x, _mm256_cmp_ps(x, zeros, _CMP_NLE_UQ)),
            _mm256_and_ps(
                _mm256_mul_ps(fills, _mm256_sub_ps(_mm256_exp_ps(x), ones)),
                _mm256_cmp_ps(x, zeros, _CMP_LT_OS)
            )
        );

        _mm256_maskstore_ps(dst_ptr + j, mask, y);
    }
}

void TensorShaderAvxBackend::Elementwise::Elu(unsigned int index, unsigned int length, float slope, cli::array<float>^ src, cli::array<float>^ dst) {
    
    Util::CheckOutOfRange(index, length, src, dst);
    
    if (length == 1) {
        dst[index] = (src[index] > 0) ? src[index] : slope * (float)(Math::Exp(src[index]) - 1);
        return;
    }

    pin_ptr<float> pinptr_src = &src[0];
    pin_ptr<float> pinptr_dst = &dst[0];

    float* src_ptr = pinptr_src;
    float* dst_ptr = pinptr_dst;

    elu(length, slope, src_ptr + index, dst_ptr + index);
}