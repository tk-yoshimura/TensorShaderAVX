#include "../../TensorShaderAvxBackend.h"

using namespace System;

__forceinline __m128 _mm_trivectornormalizegrad_ps(__m128 v, __m128 g) {

    __m128 vv_xyz = _mm_mul_ps(v, v);
    __m128 vv_yzx = _mm_permute_ps(vv_xyz, _MM_PERM_DACB);
    __m128 vv_zxy = _mm_permute_ps(vv_xyz, _MM_PERM_DBAC);

    __m128 vg_xyz = _mm_mul_ps(v, g);
    __m128 vg_yzx = _mm_permute_ps(vg_xyz, _MM_PERM_DACB);
    __m128 vg_zxy = _mm_permute_ps(vg_xyz, _MM_PERM_DBAC);

    __m128 length = _mm_sqrt_ps(_mm_dp_ps(v, v, 119));

    __m128 length_cube = _mm_max_ps(_mm_mul_ps(length, _mm_mul_ps(length, length)), _mm_set1_ps((float)1e-8));

    __m128 vv_sum = _mm_add_ps(vv_yzx, vv_zxy);
    __m128 vg_sum = _mm_add_ps(vg_yzx, vg_zxy);

    __m128 a = _mm_fmsub_ps(g, vv_sum, _mm_mul_ps(v, vg_sum));
    
    __m128 u = _mm_div_ps(a, length_cube);

    return u;
}

void trivector_normalizegrad(unsigned int length, float* src1_ptr, float* src2_ptr, float* dst_ptr) {
    const __m128i mask3 = TensorShaderAvxBackend::masktable_m128(3);

    for (unsigned int i = 0; i < length; i += 3) {
        __m128 x1 = _mm_maskload_ps(src1_ptr + i, mask3);
        __m128 x2 = _mm_maskload_ps(src2_ptr + i, mask3);

        __m128 y = _mm_trivectornormalizegrad_ps(x2, x1);

        _mm_maskstore_ps(dst_ptr + i, mask3, y);
    }
}

void TensorShaderAvxBackend::Trivector::NormalizeGrad(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst) {

    Util::CheckOutOfRange(index, length, src1, src2, dst);

    if (length % 3 != 0) {
        throw gcnew System::ArgumentException();
    }

    pin_ptr<float> pinptr_src1 = &src1[0];
    pin_ptr<float> pinptr_src2 = &src2[0];
    pin_ptr<float> pinptr_dst = &dst[0];

    float* src1_ptr = pinptr_src1;
    float* src2_ptr = pinptr_src2;
    float* dst_ptr = pinptr_dst;

    trivector_normalizegrad(length, src1_ptr + index, src2_ptr + index, dst_ptr + index);
}
