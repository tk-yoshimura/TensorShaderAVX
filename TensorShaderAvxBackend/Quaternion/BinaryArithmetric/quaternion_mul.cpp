#include "../../TensorShaderAvxBackend.h"

using namespace System;

__forceinline __m256 _mm256_quaternionmul_ps(__m256 u, __m256 v) {
    const __m256 sign_mpmp = _mm256_castsi256_ps(_mm256_setr_epi32(0x80000000ul, 0, 0x80000000ul, 0, 0x80000000ul, 0, 0x80000000ul, 0));
    const __m256 sign_mppm = _mm256_castsi256_ps(_mm256_setr_epi32(0x80000000ul, 0, 0, 0x80000000ul, 0x80000000ul, 0, 0, 0x80000000ul));
    const __m256 sign_mmpp = _mm256_castsi256_ps(_mm256_setr_epi32(0x80000000ul, 0x80000000ul, 0, 0, 0x80000000ul, 0x80000000ul, 0, 0));

    __m256 u_xxxx = _mm256_permute_ps(u, _MM_PERM_AAAA);
    __m256 u_yyyy = _mm256_permute_ps(u, _MM_PERM_BBBB);
    __m256 u_zzzz = _mm256_permute_ps(u, _MM_PERM_CCCC);
    __m256 u_wwww = _mm256_permute_ps(u, _MM_PERM_DDDD);

    __m256 v_xyzw = v;
    __m256 v_yxwz = _mm256_xor_ps(sign_mpmp, _mm256_permute_ps(v_xyzw, _MM_PERM_CDAB));
    __m256 v_zwxy = _mm256_xor_ps(sign_mppm, _mm256_permute_ps(v_xyzw, _MM_PERM_BADC));
    __m256 v_wzyx = _mm256_xor_ps(sign_mmpp, _mm256_permute_ps(v_xyzw, _MM_PERM_ABCD));

    return _mm256_fmadd_ps(u_xxxx, v_xyzw, _mm256_fmadd_ps(u_yyyy, v_yxwz, _mm256_fmadd_ps(u_zzzz, v_zwxy, _mm256_mul_ps(u_wwww, v_wzyx))));
}

void quaternion_mul(unsigned int length, float* src1_ptr, float* src2_ptr, float* dst_ptr) {
    const unsigned int j = length & ~7u, k = length - j;

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x1 = _mm256_loadu_ps(src1_ptr + i);
        __m256 x2 = _mm256_loadu_ps(src2_ptr + i);

        __m256 y = _mm256_quaternionmul_ps(x1, x2);

        _mm256_storeu_ps(dst_ptr + i, y);
    }

    if (k > 0) {
        __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        __m256 x1 = _mm256_maskload_ps(src1_ptr + j, mask);
        __m256 x2 = _mm256_maskload_ps(src2_ptr + j, mask);

        __m256 y = _mm256_quaternionmul_ps(x1, x2);

        _mm256_maskstore_ps(dst_ptr + j, mask, y);
    }
}

void TensorShaderAvxBackend::Quaternion::Mul(unsigned int index, unsigned int length, cli::array<float>^ src1, cli::array<float>^ src2, cli::array<float>^ dst) {

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

    quaternion_mul(length, src1_ptr + index, src2_ptr + index, dst_ptr + index);
}