#include "../../TensorShaderAvxBackend.h"

using namespace System;

__forceinline __m256 _mm256_leakyrelugrad_ps(__m256 x1, __m256 x2, float slope) {
    __m256 zeros = _mm256_setzero_ps();
    __m256 ones = _mm256_set1_ps(1);
    __m256 fills = _mm256_set1_ps(slope);

    __m256 y = _mm256_or_ps(
        _mm256_and_ps(x1, _mm256_cmp_ps(x2, zeros, _CMP_GT_OS)),
        _mm256_and_ps(_mm256_mul_ps(x1, fills), _mm256_cmp_ps(x2, zeros, _CMP_NGE_UQ))
    );

    return y;
}

void leakyrelugrad(unsigned int length, float slope, float* src1_ptr, float* src2_ptr, float* __restrict dst_ptr) {
    const unsigned int j = length & ~7u, k = length - j;
    
    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x1 = _mm256_load_ps(src1_ptr + i);
        __m256 x2 = _mm256_load_ps(src2_ptr + i);

        __m256 y = _mm256_leakyrelugrad_ps(x1, x2, slope);

        _mm256_store_ps(dst_ptr + i, y);
    }

    if (k > 0) {
        __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        __m256 x1 = _mm256_maskload_ps(src1_ptr + j, mask);
        __m256 x2 = _mm256_maskload_ps(src2_ptr + j, mask);

        __m256 y = _mm256_leakyrelugrad_ps(x1, x2, slope);

        _mm256_maskstore_ps(dst_ptr + j, mask, y);
    }
}

void TensorShaderAvxBackend::Elementwise::LeakyReluGrad(unsigned int length, float slope, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst) {
    
    Util::CheckLength(length, src1, src2, dst);
    
    float* src1_ptr = (float*)(src1->Ptr.ToPointer());
    float* src2_ptr = (float*)(src2->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    leakyrelugrad(length, slope, src1_ptr, src2_ptr, dst_ptr);
}
