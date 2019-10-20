#include "../../TensorShaderAvxBackend.h"

using namespace System;

__forceinline __m256 _mm256_quaternionsquashgrad_ps(__m256 v, __m256 g) {

    __m256 vv = _mm256_mul_ps(v, v);
    __m256 vg_xyzw = _mm256_mul_ps(v, g);
    __m256 vg_yzwx = _mm256_permute_ps(vg_xyzw, _MM_PERM_ADCB);
    __m256 vg_zwxy = _mm256_permute_ps(vg_xyzw, _MM_PERM_BADC);
    __m256 vg_wxyz = _mm256_permute_ps(vg_xyzw, _MM_PERM_CBAD);

    __m256 length = _mm256_sqrt_ps(_mm256_dp_ps(v, v, 255));
    __m256 length_p1 = _mm256_add_ps(length, _mm256_set1_ps(1));

    __m256 length_length_p1 = _mm256_mul_ps(length, length_p1);
    __m256 length_length_p1_length_p1 = _mm256_mul_ps(length_length_p1, length_p1);

    __m256 vg_sum = _mm256_add_ps(vg_wxyz, _mm256_add_ps(vg_yzwx, vg_zwxy));

    __m256 a = _mm256_fmsub_ps(g, _mm256_sub_ps(length_length_p1, vv), _mm256_mul_ps(v, vg_sum));

    __m256 u = _mm256_div_ps(a, length_length_p1_length_p1);

    return u;
}

void quaternion_squashgrad(unsigned int length, float* src1_ptr, float* src2_ptr, float* dst_ptr) {
    const unsigned int j = length & ~7u, k = length - j;

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x1 = _mm256_loadu_ps(src1_ptr + i);
        __m256 x2 = _mm256_loadu_ps(src2_ptr + i);

        __m256 y = _mm256_quaternionsquashgrad_ps(x2, x1);

        _mm256_storeu_ps(dst_ptr + i, y);
    }

    if (k > 0) {
        __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        __m256 x1 = _mm256_maskload_ps(src1_ptr + j, mask);
        __m256 x2 = _mm256_maskload_ps(src2_ptr + j, mask);

        __m256 y = _mm256_quaternionsquashgrad_ps(x2, x1);

        _mm256_maskstore_ps(dst_ptr + j, mask, y);
    }
}

void TensorShaderAvxBackend::Quaternion::SquashGrad(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst) {

    Util::CheckOutOfRange(index, length, src1, src2, dst);

    if (length % 4 != 0) {
        throw gcnew System::ArgumentException();
    }

    pin_ptr<float> pinptr_src1 = &src1[0];
    pin_ptr<float> pinptr_src2 = &src2[0];
    pin_ptr<float> pinptr_dst = &dst[0];

    float* src1_ptr = pinptr_src1;
    float* src2_ptr = pinptr_src2;
    float* dst_ptr = pinptr_dst;

    quaternion_squashgrad(length, src1_ptr + index, src2_ptr + index, dst_ptr + index);
}
