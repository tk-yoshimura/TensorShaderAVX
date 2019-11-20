#include "../../TensorShaderAvxBackend.h"

using namespace System;

__forceinline __m128 _mm_trivectordecay_ps(__m128 u) {
    __m128 u_norm = _mm_dp_ps(u, u, 119);
    __m128 u_mul = _mm_div_ps(u_norm, _mm_add_ps(u_norm, _mm_set1_ps(1)));

    __m128 v = _mm_mul_ps(u, u_mul);

    return v;
}

void trivector_decay(unsigned int length, float* src_ptr, float* dst_ptr) {
    const __m128i mask3 = TensorShaderAvxBackend::masktable_m128(3);

    for (unsigned int i = 0; i < length; i += 3) {
        __m128 x = _mm_maskload_ps(src_ptr + i, mask3);

        __m128 y = _mm_trivectordecay_ps(x);

        _mm_maskstore_ps(dst_ptr + i, mask3, y);
    }
}

void TensorShaderAvxBackend::Trivector::Decay(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst) {

    Util::CheckLength(length, src, dst);

    if (length % 3 != 0) {
        throw gcnew System::ArgumentException();
    }

    float* src_ptr = (float*)(src->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    trivector_decay(length, src_ptr, dst_ptr);
}