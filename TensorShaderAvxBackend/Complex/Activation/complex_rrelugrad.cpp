#include "../../TensorShaderAvxBackend.h"

#include <limits>

using namespace System;

__forceinline __m256 _mm256_complex_rrelugrad_ps(__m256 x1, __m256 x2) {
    const __m256 zeromins = _mm256_setr_ps(0, -HUGE_VALF, 0, -HUGE_VALF, 0, -HUGE_VALF, 0, -HUGE_VALF);

    __m256 y = _mm256_and_ps(x1, _mm256_cmp_ps(x2, zeromins, _CMP_GT_OS));

    return y;
}

void complex_rrelugrad(unsigned int length, float* src1_ptr, float* src2_ptr, float* __restrict dst_ptr) {
    const unsigned int j = length & ~7u, k = length - j;

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x1 = _mm256_load_ps(src1_ptr + i);
        __m256 x2 = _mm256_load_ps(src2_ptr + i);

        __m256 y = _mm256_complex_rrelugrad_ps(x1, x2);

        _mm256_store_ps(dst_ptr + i, y);
    }

    if (k > 0) {
        __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        __m256 x1 = _mm256_maskload_ps(src1_ptr + j, mask);
        __m256 x2 = _mm256_maskload_ps(src2_ptr + j, mask);

        __m256 y = _mm256_complex_rrelugrad_ps(x1, x2);

        _mm256_maskstore_ps(dst_ptr + j, mask, y);
    }
}

void TensorShaderAvxBackend::Complex::RReluGrad(unsigned int length, AvxArray<float>^ src1, AvxArray<float>^ src2, AvxArray<float>^ dst) {

    Util::CheckLength(length, src1, src2, dst);

    if (length % 2 != 0) {
        throw gcnew System::ArgumentException();
    }

    float* src1_ptr = (float*)(src1->Ptr.ToPointer());
    float* src2_ptr = (float*)(src2->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    complex_rrelugrad(length, src1_ptr, src2_ptr, dst_ptr);
}
