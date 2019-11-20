#include "../../TensorShaderAvxBackend.h"

#include <limits>

using namespace System;

__forceinline __m256 _mm256_complex_rrelu_ps(__m256 x) {
    const __m256 zeromins = _mm256_setr_ps(0, -HUGE_VALF, 0, -HUGE_VALF, 0, -HUGE_VALF, 0, -HUGE_VALF);

    __m256 y = _mm256_max_ps(zeromins, x);

    return y;
}

void complex_rrelu(unsigned int length, float* src_ptr, float* dst_ptr) {
    const unsigned int j = length & ~7u, k = length - j;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x = _mm256_load_ps(src_ptr + i);

        __m256 y = _mm256_complex_rrelu_ps(x);

        _mm256_store_ps(dst_ptr + i, y);
    }

    if (k > 0) {
        __m256 x = _mm256_maskload_ps(src_ptr + j, mask);

        __m256 y = _mm256_complex_rrelu_ps(x);

        _mm256_maskstore_ps(dst_ptr + j, mask, y);
    }
}

void TensorShaderAvxBackend::Complex::RRelu(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst) {

    Util::CheckLength(length, src, dst);

    if (length % 2 != 0) {
        throw gcnew System::ArgumentException();
    }
    
    float* src_ptr = (float*)(src->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    complex_rrelu(length, src_ptr, dst_ptr);
}