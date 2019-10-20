#include "../../TensorShaderAvxBackend.h"

#include <limits>

using namespace System;

void complex_rrelugrad(unsigned int length, float* src1_ptr, float* src2_ptr, float* dst_ptr) {
    const unsigned int j = length & ~7u, k = length - j;

    const __m256 zeromins = _mm256_setr_ps(0, -HUGE_VALF, 0, -HUGE_VALF, 0, -HUGE_VALF, 0, -HUGE_VALF);

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x1 = _mm256_loadu_ps(src1_ptr + i);
        __m256 x2 = _mm256_loadu_ps(src2_ptr + i);

        __m256 y = _mm256_and_ps(x1, _mm256_cmp_ps(x2, zeromins, _CMP_GT_OS));

        _mm256_storeu_ps(dst_ptr + i, y);
    }

    if (k > 0) {
        __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        __m256 x1 = _mm256_maskload_ps(src1_ptr + j, mask);
        __m256 x2 = _mm256_maskload_ps(src2_ptr + j, mask);

        __m256 y = _mm256_and_ps(x1, _mm256_cmp_ps(x2, zeromins, _CMP_GT_OS));

        _mm256_maskstore_ps(dst_ptr + j, mask, y);
    }
}

void TensorShaderAvxBackend::Complex::RReluGrad(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst) {

    Util::CheckOutOfRange(index, length, src1, src2, dst);

    if (length % 2 != 0) {
        throw gcnew System::ArgumentException();
    }

    pin_ptr<float> pinptr_src1 = &src1[0];
    pin_ptr<float> pinptr_src2 = &src2[0];
    pin_ptr<float> pinptr_dst = &dst[0];

    float* src1_ptr = pinptr_src1;
    float* src2_ptr = pinptr_src2;
    float* dst_ptr = pinptr_dst;

    complex_rrelugrad(length, src1_ptr + index, src2_ptr + index, dst_ptr + index);
}
