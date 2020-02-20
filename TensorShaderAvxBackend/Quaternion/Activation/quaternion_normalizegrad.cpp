#include "../../TensorShaderAvxBackend.h"

using namespace System;

__forceinline __m256 _mm256_quaternionnormalizegrad_ps(__m256 v, __m256 g) {

    __m256 vv_xyzw = _mm256_mul_ps(v, v);
    __m256 vv_yzwx = _mm256_permute_ps(vv_xyzw, _MM_PERM_ADCB);
    __m256 vv_zwxy = _mm256_permute_ps(vv_xyzw, _MM_PERM_BADC);
    __m256 vv_wxyz = _mm256_permute_ps(vv_xyzw, _MM_PERM_CBAD);

    __m256 vg_xyzw = _mm256_mul_ps(v, g);
    __m256 vg_yzwx = _mm256_permute_ps(vg_xyzw, _MM_PERM_ADCB);
    __m256 vg_zwxy = _mm256_permute_ps(vg_xyzw, _MM_PERM_BADC);
    __m256 vg_wxyz = _mm256_permute_ps(vg_xyzw, _MM_PERM_CBAD);

    __m256 length = _mm256_sqrt_ps(_mm256_dp_ps(v, v, 255));

    __m256 length_cube = _mm256_max_ps(_mm256_mul_ps(length, _mm256_mul_ps(length, length)), _mm256_set1_ps((float)1e-8));

    __m256 vv_sum = _mm256_add_ps(vv_wxyz, _mm256_add_ps(vv_yzwx, vv_zwxy));
    __m256 vg_sum = _mm256_add_ps(vg_wxyz, _mm256_add_ps(vg_yzwx, vg_zwxy));

    __m256 a = _mm256_fmsub_ps(g, vv_sum, _mm256_mul_ps(v, vg_sum));

    __m256 u = _mm256_div_ps(a, length_cube);

    return u;
}

void quaternion_normalizegrad(unsigned int length, float* src1_ptr, float* src2_ptr, float* __restrict dst_ptr) {
    const unsigned int j = length & ~7u, k = length - j;

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x1 = _mm256_load_ps(src1_ptr + i);
        __m256 x2 = _mm256_load_ps(src2_ptr + i);

        __m256 y = _mm256_quaternionnormalizegrad_ps(x2, x1);

        _mm256_store_ps(dst_ptr + i, y);
    }

    if (k > 0) {
        __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        __m256 x1 = _mm256_maskload_ps(src1_ptr + j, mask);
        __m256 x2 = _mm256_maskload_ps(src2_ptr + j, mask);

        __m256 y = _mm256_quaternionnormalizegrad_ps(x2, x1);

        _mm256_maskstore_ps(dst_ptr + j, mask, y);
    }
}

void TensorShaderAvxBackend::Quaternion::NormalizeGrad(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst) {

    Util::CheckLength(length, src1, src2, dst);

    if (length % 4 != 0) {
        throw gcnew System::ArgumentException();
    }

    float* src1_ptr = (float*)(src1->Ptr.ToPointer());
    float* src2_ptr = (float*)(src2->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    quaternion_normalizegrad(length, src1_ptr, src2_ptr, dst_ptr);
}
