#include "../../TensorShaderAvxBackend.h"

using namespace System;

__forceinline __m128 _mm_trivectorcross_ps(__m128 u, __m128 v) {
    __m128 u_yzx = _mm_permute_ps(u, _MM_PERM_DACB);
    __m128 v_yzx = _mm_permute_ps(v, _MM_PERM_DACB);

    __m128 u_zxy = _mm_permute_ps(u, _MM_PERM_DBAC);
    __m128 v_zxy = _mm_permute_ps(v, _MM_PERM_DBAC);

    return _mm_fmsub_ps(u_yzx, v_zxy, _mm_mul_ps(u_zxy, v_yzx));
}

void trivector_cross(unsigned int length, float* src1_ptr, float* src2_ptr, float* __restrict dst_ptr) {
    const __m128i mask3 = TensorShaderAvxBackend::masktable_m128(3);

    for (unsigned int i = 0; i < length; i += 3) {
        __m128 x1 = _mm_maskload_ps(src1_ptr + i, mask3);
        __m128 x2 = _mm_maskload_ps(src2_ptr + i, mask3);

        __m128 y = _mm_trivectorcross_ps(x1, x2);

        _mm_maskstore_ps(dst_ptr + i, mask3, y);
    }
}

void TensorShaderAvxBackend::Trivector::Cross(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst) {

    Util::CheckLength(length, src1, src2, dst);

    if (length % 3 != 0) {
        throw gcnew System::ArgumentException();
    }

    float* src1_ptr = (float*)(src1->Ptr.ToPointer());
    float* src2_ptr = (float*)(src2->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    trivector_cross(length, src1_ptr, src2_ptr, dst_ptr);
}