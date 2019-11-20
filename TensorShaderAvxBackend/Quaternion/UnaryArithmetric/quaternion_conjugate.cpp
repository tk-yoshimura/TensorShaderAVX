#include "../../TensorShaderAvxBackend.h"

#include <limits>

using namespace System;

void quaternion_conjugate(unsigned int length, float* src_ptr, float* dst_ptr) {
    const unsigned int j = length & ~7u, k = length - j;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

    __m256 sign = _mm256_castsi256_ps(_mm256_setr_epi32(0, 0x80000000u, 0x80000000u, 0x80000000u, 0, 0x80000000u, 0x80000000u, 0x80000000u));

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x = _mm256_load_ps(src_ptr + i);

        __m256 y = _mm256_xor_ps(sign, x);

        _mm256_store_ps(dst_ptr + i, y);
    }

    if (k > 0) {
        __m256 x = _mm256_maskload_ps(src_ptr + j, mask);

        __m256 y = _mm256_xor_ps(sign, x);

        _mm256_maskstore_ps(dst_ptr + j, mask, y);
    }
}

void TensorShaderAvxBackend::Quaternion::Conjugate(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst) {

    Util::CheckLength(length, src, dst);

    if (length % 4 != 0) {
        throw gcnew System::ArgumentException();
    }

    float* src_ptr = (float*)(src->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    quaternion_conjugate(length, src_ptr, dst_ptr);
}