#include "../../TensorShaderAvxBackend.h"

using namespace System;

__forceinline __m128 _mm_trivectornormalize_ps(__m128 u) {
    __m128 u_norm = _mm_dp_ps(u, u, 119);
    __m128 u_length = _mm_max_ps(_mm_sqrt_ps(u_norm), _mm_set1_ps((float)1e-8));

    __m128 v = _mm_div_ps(u, u_length);

    return v;
}

void trivector_normalize(unsigned int length, const float* __restrict src_ptr, float* __restrict dst_ptr) {
    const __m128i mask3 = TensorShaderAvxBackend::masktable_m128(3);

    for (unsigned int i = 0; i < length; i += 3) {
        __m128 x = _mm_maskload_ps(src_ptr + i, mask3);

        __m128 y = _mm_trivectornormalize_ps(x);

        _mm_maskstore_ps(dst_ptr + i, mask3, y);
    }
}

void TensorShaderAvxBackend::Trivector::Normalize(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst) {

    Util::CheckLength(length, src, dst);

    if (length % 3 != 0) {
        throw gcnew System::ArgumentException();
    }

    const float* src_ptr = (const float*)(src->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    trivector_normalize(length, src_ptr, dst_ptr);
}