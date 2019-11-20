#include "../../TensorShaderAvxBackend.h"

#include <limits>

using namespace System;

__forceinline __m256 _mm256_complexsquare_ps(__m256 u) {
    __m256 uri = u;
    __m256 urr = _mm256_permute_ps(u, _MM_PERM_CCAA);
    __m256 uir = _mm256_permute_ps(u, _MM_PERM_CDAB);
    __m256 uii = _mm256_permute_ps(u, _MM_PERM_DDBB);

    return _mm256_fmaddsub_ps(uri, urr, _mm256_mul_ps(uir, uii));
}

void complex_square(unsigned int length, float* src_ptr, float* dst_ptr) {
    const unsigned int j = length & ~7u, k = length - j;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);
    
    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x = _mm256_load_ps(src_ptr + i);

        __m256 y = _mm256_complexsquare_ps(x);

        _mm256_store_ps(dst_ptr + i, y);
    }

    if (k > 0) {
        __m256 x = _mm256_maskload_ps(src_ptr + j, mask);

        __m256 y = _mm256_complexsquare_ps(x);

        _mm256_maskstore_ps(dst_ptr + j, mask, y);
    }
}

void TensorShaderAvxBackend::Complex::Square(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst) {

    Util::CheckLength(length, src, dst);

    if (length % 2 != 0) {
        throw gcnew System::ArgumentException();
    }

    float* src_ptr = (float*)(src->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    complex_square(length, src_ptr, dst_ptr);
}