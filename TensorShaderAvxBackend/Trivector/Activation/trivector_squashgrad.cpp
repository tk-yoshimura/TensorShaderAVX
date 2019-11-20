#include "../../TensorShaderAvxBackend.h"

using namespace System;

__forceinline __m128 _mm_trivectorsquashgrad_ps(__m128 v, __m128 g) {

    __m128 vv = _mm_mul_ps(v, v);
    __m128 vg_xyz = _mm_mul_ps(v, g);
    __m128 vg_yzx = _mm_permute_ps(vg_xyz, _MM_PERM_DACB);
    __m128 vg_zxy = _mm_permute_ps(vg_xyz, _MM_PERM_DBAC);

    __m128 length = _mm_sqrt_ps(_mm_dp_ps(v, v, 119));
    __m128 length_p1 = _mm_add_ps(length, _mm_set1_ps(1));

    __m128 length_length_p1 = _mm_mul_ps(length, length_p1);
    __m128 length_length_p1_length_p1 = _mm_mul_ps(length_length_p1, length_p1);

    __m128 vg_sum = _mm_add_ps(vg_yzx, vg_zxy);

    __m128 a = _mm_fmsub_ps(g, _mm_sub_ps(length_length_p1, vv), _mm_mul_ps(v, vg_sum));

    __m128 u = _mm_div_ps(a, length_length_p1_length_p1);

    return u;
}

void trivector_squashgrad(unsigned int length, float* src1_ptr, float* src2_ptr, float* dst_ptr) {
    const __m128i mask3 = TensorShaderAvxBackend::masktable_m128(3);

    for (unsigned int i = 0; i < length; i += 3) {
        __m128 x1 = _mm_maskload_ps(src1_ptr + i, mask3);
        __m128 x2 = _mm_maskload_ps(src2_ptr + i, mask3);

        __m128 y = _mm_trivectorsquashgrad_ps(x2, x1);

        _mm_maskstore_ps(dst_ptr + i, mask3, y);
    }
}

void TensorShaderAvxBackend::Trivector::SquashGrad(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst) {

    Util::CheckLength(length, src1, src2, dst);

    if (length % 3 != 0) {
        throw gcnew System::ArgumentException();
    }

    float* src1_ptr = (float*)(src1->Ptr.ToPointer());
    float* src2_ptr = (float*)(src2->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    trivector_squashgrad(length, src1_ptr, src2_ptr, dst_ptr);
}
