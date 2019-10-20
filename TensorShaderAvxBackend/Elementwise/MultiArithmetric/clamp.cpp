#include "../../TensorShaderAvxBackend.h"

using namespace System;

void clamp(unsigned int length, float* srcval_ptr, float* srcmin_ptr, float* srcmax_ptr, float* dst_ptr) {
    const unsigned int j = length & ~7u, k = length - j;

    __m256 zeros = _mm256_setzero_ps();

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 xval = _mm256_loadu_ps(srcval_ptr + i);
        __m256 xmin = _mm256_loadu_ps(srcmin_ptr + i);
        __m256 xmax = _mm256_loadu_ps(srcmax_ptr + i);

        __m256 y = _mm256_min_ps(_mm256_max_ps(xval, xmin), xmax);

        _mm256_storeu_ps(dst_ptr + i, y);
    }

    if (k > 0) {
        __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        __m256 xval = _mm256_maskload_ps(srcval_ptr + j, mask);
        __m256 xmin = _mm256_maskload_ps(srcmin_ptr + j, mask);
        __m256 xmax = _mm256_maskload_ps(srcmax_ptr + j, mask);

        __m256 y = _mm256_min_ps(_mm256_max_ps(xval, xmin), xmax);

        _mm256_maskstore_ps(dst_ptr + j, mask, y);
    }
}

void TensorShaderAvxBackend::Elementwise::Clamp(unsigned int index, unsigned int length, cli::array<float>^ srcval, cli::array<float>^ srcmin, cli::array<float>^ srcmax, cli::array<float>^ dst) {
    
    Util::CheckOutOfRange(index, length, srcval, srcmin, srcmax, dst);
    
    if (length == 1) {
        dst[index] = Math::Min(Math::Max(srcval[index], srcmin[index]), srcmax[index]);
        return;
    }

    pin_ptr<float> pinptr_srcval = &srcval[0];
    pin_ptr<float> pinptr_srcmin = &srcmin[0];
    pin_ptr<float> pinptr_srcmax = &srcmax[0];
    pin_ptr<float> pinptr_dst = &dst[0];

    float* srcval_ptr = pinptr_srcval;
    float* srcmin_ptr = pinptr_srcmin;
    float* srcmax_ptr = pinptr_srcmax;
    float* dst_ptr = pinptr_dst;

    clamp(length, srcval_ptr + index, srcmin_ptr + index, srcmax_ptr + index, dst_ptr + index);
}
