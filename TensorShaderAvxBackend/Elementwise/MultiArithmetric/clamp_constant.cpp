#include "../../TensorShaderAvxBackend.h"

using namespace System;

void clamp_constant(unsigned int length, float cmin, float cmax, float* src_ptr, float* dst_ptr) {
    const unsigned int j = length & ~7u, k = length - j;

    __m256 fillcmin = _mm256_set1_ps(cmin), fillcmax = _mm256_set1_ps(cmax);

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x = _mm256_loadu_ps(src_ptr + i);

        __m256 y = _mm256_min_ps(_mm256_max_ps(x, fillcmin), fillcmax);

        _mm256_storeu_ps(dst_ptr + i, y);
    }

    if (k > 0) {
        __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        __m256 x = _mm256_maskload_ps(src_ptr + j, mask);

        __m256 y = _mm256_min_ps(_mm256_max_ps(x, fillcmin), fillcmax);

        _mm256_maskstore_ps(dst_ptr + j, mask, y);
    }
}

void TensorShaderAvxBackend::Elementwise::ClampConstant(unsigned int index, unsigned int length, float cmin, float cmax, cli::array<float>^ src, cli::array<float>^ dst) {
    
    Util::CheckOutOfRange(index, length, src, dst);
    
    if (length == 1) {
        dst[index] = Math::Min(Math::Max(src[index], cmin), cmax);
        return;
    }

    pin_ptr<float> pinptr_src = &src[0];
    pin_ptr<float> pinptr_dst = &dst[0];

    float* src_ptr = pinptr_src;
    float* dst_ptr = pinptr_dst;

    clamp_constant(length, cmin, cmax, src_ptr + index, dst_ptr + index);
}