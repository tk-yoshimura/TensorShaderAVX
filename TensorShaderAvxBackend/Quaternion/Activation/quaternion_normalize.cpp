#include "../../TensorShaderAvxBackend.h"

using namespace System;

__forceinline __m256 _mm256_quaternionnormalize_ps(__m256 u) {
    __m256 u_norm = _mm256_dp_ps(u, u, 255);
    __m256 u_length = _mm256_max_ps(_mm256_sqrt_ps(u_norm), _mm256_set1_ps((float)1e-8));

    __m256 v = _mm256_div_ps(u, u_length);

    return v;
}

void quaternion_normalize(unsigned int length, float* src_ptr, float* dst_ptr) {
    const unsigned int j = length & ~7u, k = length - j;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x = _mm256_load_ps(src_ptr + i);

        __m256 y = _mm256_quaternionnormalize_ps(x);

        _mm256_store_ps(dst_ptr + i, y);
    }

    if (k > 0) {
        __m256 x = _mm256_maskload_ps(src_ptr + j, mask);

        __m256 y = _mm256_quaternionnormalize_ps(x);

        _mm256_maskstore_ps(dst_ptr + j, mask, y);
    }
}

void TensorShaderAvxBackend::Quaternion::Normalize(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst) {

    Util::CheckLength(length, src, dst);

    if (length % 4 != 0) {
        throw gcnew System::ArgumentException();
    }
    
    float* src_ptr = (float*)(src->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    quaternion_normalize(length, src_ptr, dst_ptr);
}