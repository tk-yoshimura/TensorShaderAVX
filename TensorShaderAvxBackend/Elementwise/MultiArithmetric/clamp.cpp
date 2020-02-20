#include "../../TensorShaderAvxBackend.h"

using namespace System;

__forceinline __m256 _mm256_clamp_ps(__m256 xval, __m256 xmin, __m256 xmax) {
    __m256 y = _mm256_min_ps(_mm256_max_ps(xval, xmin), xmax);

    return y;
}

void clamp(unsigned int length, float* srcval_ptr, float* srcmin_ptr, float* srcmax_ptr, float* __restrict dst_ptr) {
    const unsigned int j = length & ~7u, k = length - j;
    
    for (unsigned int i = 0; i < j; i += 8) {
        __m256 xval = _mm256_load_ps(srcval_ptr + i);
        __m256 xmin = _mm256_load_ps(srcmin_ptr + i);
        __m256 xmax = _mm256_load_ps(srcmax_ptr + i);

        __m256 y = _mm256_clamp_ps(xval, xmin, xmax);

        _mm256_store_ps(dst_ptr + i, y);
    }

    if (k > 0) {
        __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        __m256 xval = _mm256_maskload_ps(srcval_ptr + j, mask);
        __m256 xmin = _mm256_maskload_ps(srcmin_ptr + j, mask);
        __m256 xmax = _mm256_maskload_ps(srcmax_ptr + j, mask);

        __m256 y = _mm256_clamp_ps(xval, xmin, xmax);

        _mm256_maskstore_ps(dst_ptr + j, mask, y);
    }
}

void TensorShaderAvxBackend::Elementwise::Clamp(unsigned int length, AvxArray<float>^ srcval, AvxArray<float>^ srcmin, AvxArray<float>^ srcmax, AvxArray<float>^ dst) {
    
    Util::CheckLength(length, srcval, srcmin, srcmax, dst);
    
    float* srcval_ptr = (float*)(srcval->Ptr.ToPointer());
    float* srcmin_ptr = (float*)(srcmin->Ptr.ToPointer());
    float* srcmax_ptr = (float*)(srcmax->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    clamp(length, srcval_ptr, srcmin_ptr, srcmax_ptr, dst_ptr);
}
