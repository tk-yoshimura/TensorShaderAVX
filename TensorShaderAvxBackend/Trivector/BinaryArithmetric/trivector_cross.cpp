#include "../../TensorShaderAvxBackend.h"

using namespace System;

__forceinline __m128 _mm_trivectorcross_ps(__m128 u, __m128 v) {
    __m128 u_yzx = _mm_permute_ps(u, _MM_PERM_DACB);
    __m128 v_yzx = _mm_permute_ps(v, _MM_PERM_DACB);

    __m128 u_zxy = _mm_permute_ps(u, _MM_PERM_DBAC);
    __m128 v_zxy = _mm_permute_ps(v, _MM_PERM_DBAC);

    return _mm_fmsub_ps(u_yzx, v_zxy, _mm_mul_ps(u_zxy, v_yzx));
}

void trivector_cross(unsigned int length, float* src1_ptr, float* src2_ptr, float* dst_ptr) {
    const __m128i mask3 = TensorShaderAvxBackend::masktable_m128(3);

    for (unsigned int i = 0; i < length; i += 3) {
        __m128 x1 = _mm_maskload_ps(src1_ptr + i, mask3);
        __m128 x2 = _mm_maskload_ps(src2_ptr + i, mask3);

        __m128 y = _mm_trivectorcross_ps(x1, x2);

        _mm_maskstore_ps(dst_ptr + i, mask3, y);
    }
}

void TensorShaderAvxBackend::Trivector::Cross(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst) {

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

    trivector_cross(length, src1_ptr + index, src2_ptr + index, dst_ptr + index);
}