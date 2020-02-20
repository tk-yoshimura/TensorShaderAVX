#include "../../TensorShaderAvxBackend.h"

using namespace System;

__forceinline __m256 _mm256_elu_ps(__m256 x, float slope) {
    __m256 zeros = _mm256_setzero_ps();
    __m256 ones = _mm256_set1_ps(1);
    __m256 fills = _mm256_set1_ps(slope);

    __m256 y = _mm256_or_ps(
        _mm256_and_ps(x, _mm256_cmp_ps(x, zeros, _CMP_NLE_UQ)),
        _mm256_and_ps(
            _mm256_mul_ps(fills, _mm256_sub_ps(_mm256_exp_ps(x), ones)),
            _mm256_cmp_ps(x, zeros, _CMP_LT_OS)
        )
    );

    return y;
}

void elu(unsigned int length, float slope, const float* __restrict src_ptr, float* __restrict dst_ptr) {
    const unsigned int j = length & ~7u, k = length - j;

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x = _mm256_load_ps(src_ptr + i);

        __m256 y = _mm256_elu_ps(x, slope);

        _mm256_store_ps(dst_ptr + i, y);
    }

    if (k > 0) {
        __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        __m256 x = _mm256_maskload_ps(src_ptr + j, mask);

        __m256 y = _mm256_elu_ps(x, slope);

        _mm256_maskstore_ps(dst_ptr + j, mask, y);
    }
}

void TensorShaderAvxBackend::Elementwise::Elu(unsigned int length, float slope, AvxArray<float>^ src, AvxArray<float>^ dst) {
    
    Util::CheckLength(length, src, dst);
    
    float* src_ptr = (float*)(src->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    elu(length, slope, src_ptr, dst_ptr);
}