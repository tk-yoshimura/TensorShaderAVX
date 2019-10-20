#include "../../TensorShaderAvxBackend.h"

#include <limits>

using namespace System;

void complex_zrelu(unsigned int length, float* src_ptr, float* dst_ptr) {
    const unsigned int j = length & ~7u, k = length - j;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

    const __m256 zeros = _mm256_setzero_ps();

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x = _mm256_loadu_ps(src_ptr + i);

        __m256 z = _mm256_cmp_ps(x, zeros, _CMP_GT_OS);
        z = _mm256_and_ps(z, _mm256_permute_ps(z, _MM_PERM_CDAB));

        __m256 y = _mm256_and_ps(z, x);

        _mm256_storeu_ps(dst_ptr + i, y);
    }

    if (k > 0) {
        __m256 x = _mm256_maskload_ps(src_ptr + j, mask);

        __m256 z = _mm256_cmp_ps(x, zeros, _CMP_GT_OS);
        z = _mm256_and_ps(z, _mm256_permute_ps(z, _MM_PERM_CDAB));

        __m256 y = _mm256_and_ps(z, x);

        _mm256_maskstore_ps(dst_ptr + j, mask, y);
    }
}

void TensorShaderAvxBackend::Complex::ZRelu(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst) {

    Util::CheckOutOfRange(index, length, src, dst);

    if (length % 2 != 0) {
        throw gcnew System::ArgumentException();
    }

    pin_ptr<float> pinptr_src = &src[0];
    pin_ptr<float> pinptr_dst = &dst[0];

    float* src_ptr = pinptr_src;
    float* dst_ptr = pinptr_dst;

    complex_zrelu(length, src_ptr + index, dst_ptr + index);
}