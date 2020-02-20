#include "../../TensorShaderAvxBackend.h"

using namespace System;

__forceinline __m256 _mm256_squaremulgrad_ps(__m256 u, __m256 v) {
    __m256 c = _mm256_set_ps(-2, 2, -2, 2, -2, 2, -2, 2);

    __m256 uri = u;
    __m256 vrr = _mm256_permute_ps(v, _MM_PERM_CCAA);
    __m256 uir = _mm256_permute_ps(u, _MM_PERM_CDAB);
    __m256 vii = _mm256_permute_ps(v, _MM_PERM_DDBB);

    return _mm256_mul_ps(_mm256_fmsubadd_ps(uri, vrr, _mm256_mul_ps(uir, vii)), c);
}

void complex_squaregrad(unsigned int length, float* src1_ptr, float* src2_ptr, float* __restrict dst_ptr) {
    const unsigned int j = length & ~7u, k = length - j;

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x1 = _mm256_load_ps(src1_ptr + i);
        __m256 x2 = _mm256_load_ps(src2_ptr + i);

        __m256 y = _mm256_squaremulgrad_ps(x2, x1);

        _mm256_store_ps(dst_ptr + i, y);
    }

    if (k > 0) {
        __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        __m256 x1 = _mm256_maskload_ps(src1_ptr + j, mask);
        __m256 x2 = _mm256_maskload_ps(src2_ptr + j, mask);

        __m256 y = _mm256_squaremulgrad_ps(x2, x1);

        _mm256_maskstore_ps(dst_ptr + j, mask, y);
    }
}

void TensorShaderAvxBackend::Complex::SquareGrad(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst) {

    Util::CheckLength(length, src1, src2, dst);

    if (length % 2 != 0) {
        throw gcnew System::ArgumentException();
    }

    float* src1_ptr = (float*)(src1->Ptr.ToPointer());
    float* src2_ptr = (float*)(src2->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    complex_squaregrad(length, src1_ptr, src2_ptr, dst_ptr);
}
