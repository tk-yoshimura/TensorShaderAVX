#include "../../TensorShaderAvxBackend.h"

#include <limits>

using namespace System;

__forceinline __m256 _mm256_quaternionsquare_ps(__m256 u) {
    const __m256 sign_mpmp = _mm256_castsi256_ps(_mm256_setr_epi32(0x80000000ul, 0, 0x80000000ul, 0, 0x80000000ul, 0, 0x80000000ul, 0));
    const __m256 sign_mppm = _mm256_castsi256_ps(_mm256_setr_epi32(0x80000000ul, 0, 0, 0x80000000ul, 0x80000000ul, 0, 0, 0x80000000ul));
    const __m256 sign_mmpp = _mm256_castsi256_ps(_mm256_setr_epi32(0x80000000ul, 0x80000000ul, 0, 0, 0x80000000ul, 0x80000000ul, 0, 0));

    __m256 u_xxxx = _mm256_permute_ps(u, _MM_PERM_AAAA);
    __m256 u_yyyy = _mm256_permute_ps(u, _MM_PERM_BBBB);
    __m256 u_zzzz = _mm256_permute_ps(u, _MM_PERM_CCCC);
    __m256 u_wwww = _mm256_permute_ps(u, _MM_PERM_DDDD);

    __m256 u_xyzw = u;
    __m256 u_yxwz = _mm256_xor_ps(sign_mpmp, _mm256_permute_ps(u_xyzw, _MM_PERM_CDAB));
    __m256 u_zwxy = _mm256_xor_ps(sign_mppm, _mm256_permute_ps(u_xyzw, _MM_PERM_BADC));
    __m256 u_wzyx = _mm256_xor_ps(sign_mmpp, _mm256_permute_ps(u_xyzw, _MM_PERM_ABCD));

    return _mm256_fmadd_ps(u_xxxx, u_xyzw, _mm256_fmadd_ps(u_yyyy, u_yxwz, _mm256_fmadd_ps(u_zzzz, u_zwxy, _mm256_mul_ps(u_wwww, u_wzyx))));
}

void quaternion_square(unsigned int length, float* src_ptr, float* dst_ptr) {
    const unsigned int j = length & ~7u, k = length - j;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);
    
    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x = _mm256_loadu_ps(src_ptr + i);

        __m256 y = _mm256_quaternionsquare_ps(x);

        _mm256_storeu_ps(dst_ptr + i, y);
    }

    if (k > 0) {
        __m256 x = _mm256_maskload_ps(src_ptr + j, mask);

        __m256 y = _mm256_quaternionsquare_ps(x);

        _mm256_maskstore_ps(dst_ptr + j, mask, y);
    }
}

void TensorShaderAvxBackend::Quaternion::Square(unsigned int index, unsigned int length, cli::array<float>^ src, cli::array<float>^ dst) {

    Util::CheckOutOfRange(index, length, src, dst);

    if (length % 4 != 0) {
        throw gcnew System::ArgumentException();
    }

    pin_ptr<float> pinptr_src = &src[0];
    pin_ptr<float> pinptr_dst = &dst[0];

    float* src_ptr = pinptr_src;
    float* dst_ptr = pinptr_dst;

    quaternion_square(length, src_ptr + index, dst_ptr + index);
}