#include "../../TensorShaderAvxBackend.h"

using namespace System;

void elugrad(unsigned int length, float slope, float* src1_ptr, float* src2_ptr, float* dst_ptr) {
    const unsigned int j = length & ~7u, k = length - j;

    __m256 zeros = _mm256_setzero_ps();
    __m256 ones = _mm256_set1_ps(1);
    __m256 fills = _mm256_set1_ps(slope);

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x1 = _mm256_loadu_ps(src1_ptr + i);
        __m256 x2 = _mm256_loadu_ps(src2_ptr + i);

        __m256 y = _mm256_or_ps(
            _mm256_and_ps(x1, _mm256_cmp_ps(x2, zeros, _CMP_GT_OS)),
            _mm256_and_ps(
                _mm256_mul_ps(x1,
                    _mm256_mul_ps(
                        fills,
                        _mm256_exp_ps(x2)
                    )
                ),
                _mm256_cmp_ps(x2, zeros, _CMP_NGE_UQ)
            )
        );

        _mm256_storeu_ps(dst_ptr + i, y);
    }

    if (k > 0) {
        __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        __m256 x1 = _mm256_maskload_ps(src1_ptr + j, mask);
        __m256 x2 = _mm256_maskload_ps(src2_ptr + j, mask);

        __m256 y = _mm256_or_ps(
            _mm256_and_ps(x1, _mm256_cmp_ps(x2, zeros, _CMP_GT_OS)),
            _mm256_and_ps(
                _mm256_mul_ps(x1,
                    _mm256_mul_ps(
                        fills,
                        _mm256_exp_ps(x2)
                    )
                ),
                _mm256_cmp_ps(x2, zeros, _CMP_NGE_UQ)
            )
        );

        _mm256_maskstore_ps(dst_ptr + j, mask, y);
    }
}

void TensorShaderAvxBackend::Elementwise::EluGrad(unsigned int index, unsigned int length, float slope, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst) {
    
    Util::CheckOutOfRange(index, length, src1, src2, dst);
    
    if (length == 1) {
        dst[index] = (src2[index] > 0) ? src1[index] : src1[index] * (slope * (float)Math::Exp(src2[index]));
        return;
    }

    pin_ptr<float> pinptr_src1 = &src1[0];
    pin_ptr<float> pinptr_src2 = &src2[0];
    pin_ptr<float> pinptr_dst = &dst[0];

    float* src1_ptr = pinptr_src1;
    float* src2_ptr = pinptr_src2;
    float* dst_ptr = pinptr_dst;

    elugrad(length, slope, src1_ptr + index, src2_ptr + index, dst_ptr + index);
}
