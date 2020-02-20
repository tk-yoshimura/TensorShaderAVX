#include "../../TensorShaderAvxBackend.h"

using namespace System;

__forceinline __m256 _mm256_complexsquashgrad_ps(__m256 v, __m256 g) {

    __m256 vv = _mm256_mul_ps(v, v);
    __m256 vg_xy = _mm256_mul_ps(v, g);
    __m256 vg_yx = _mm256_permute_ps(vg_xy, _MM_PERM_CDAB);

    __m256 length = _mm256_sqrt_ps(_mm256_permute_ps(_mm256_hadd_ps(vv, vv), _MM_PERM_DBCA));
    __m256 length_p1 = _mm256_add_ps(length, _mm256_set1_ps(1));

    __m256 length_length_p1 = _mm256_mul_ps(length, length_p1);
    __m256 length_length_p1_length_p1 = _mm256_mul_ps(length_length_p1, length_p1);

    __m256 a = _mm256_fmsub_ps(g, _mm256_sub_ps(length_length_p1, vv), _mm256_mul_ps(v, vg_yx));

    __m256 u = _mm256_div_ps(a, length_length_p1_length_p1);

    return u;
}

void complex_squashgrad(unsigned int length, const float* __restrict src1_ptr, const float* __restrict src2_ptr, float* __restrict dst_ptr) {
    const unsigned int j = length & ~7u, k = length - j;

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x1 = _mm256_load_ps(src1_ptr + i);
        __m256 x2 = _mm256_load_ps(src2_ptr + i);

        __m256 y = _mm256_complexsquashgrad_ps(x2, x1);

        _mm256_store_ps(dst_ptr + i, y);
    }

    if (k > 0) {
        __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        __m256 x1 = _mm256_maskload_ps(src1_ptr + j, mask);
        __m256 x2 = _mm256_maskload_ps(src2_ptr + j, mask);

        __m256 y = _mm256_complexsquashgrad_ps(x2, x1);

        _mm256_maskstore_ps(dst_ptr + j, mask, y);
    }
}

void TensorShaderAvxBackend::Complex::SquashGrad(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst) {

    Util::CheckLength(length, src1, src2, dst);

    if (length % 2 != 0) {
        throw gcnew System::ArgumentException();
    }

    const float* src1_ptr = (const float*)(src1->Ptr.ToPointer());
    const float* src2_ptr = (const float*)(src2->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    complex_squashgrad(length, src1_ptr, src2_ptr, dst_ptr);
}
