#include "../../TensorShaderAvxBackend.h"

using namespace System;

__forceinline __m256 _mm256_quaterniondecay_ps(__m256 u) {
    __m256 u_norm = _mm256_dp_ps(u, u, 255);
    __m256 u_mul = _mm256_div_ps(u_norm, _mm256_add_ps(u_norm, _mm256_set1_ps(1)));

    __m256 v = _mm256_mul_ps(u, u_mul);

    return v;
}

void quaternion_decay(unsigned int length, float* src_ptr, float* dst_ptr) {
    const unsigned int j = length & ~7u, k = length - j;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x = _mm256_loadu_ps(src_ptr + i);

        __m256 y = _mm256_quaterniondecay_ps(x);

        _mm256_storeu_ps(dst_ptr + i, y);
    }

    if (k > 0) {
        __m256 x = _mm256_maskload_ps(src_ptr + j, mask);

        __m256 y = _mm256_quaterniondecay_ps(x);

        _mm256_maskstore_ps(dst_ptr + j, mask, y);
    }
}

void TensorShaderAvxBackend::Quaternion::Decay(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst) {

    Util::CheckOutOfRange(index, length, src, dst);

    if (length % 4 != 0) {
        throw gcnew System::ArgumentException();
    }

    pin_ptr<float> pinptr_src = &src[0];
    pin_ptr<float> pinptr_dst = &dst[0];

    float* src_ptr = pinptr_src;
    float* dst_ptr = pinptr_dst;

    quaternion_decay(length, src_ptr + index, dst_ptr + index);
}