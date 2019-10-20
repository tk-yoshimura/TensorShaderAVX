#include "../../TensorShaderAvxBackend.h"

using namespace System;

void insertadd_1(unsigned int length, float* src_ptr, float* ref_ptr) {
    const unsigned int j = length & ~7u, k = length - j;

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x = _mm256_loadu_ps(src_ptr + i);
        __m256 y = _mm256_loadu_ps(ref_ptr + i);

        y = _mm256_add_ps(x, y);

        _mm256_storeu_ps(ref_ptr + i, y);
    }

    if (k > 0) {
        __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        __m256 x = _mm256_maskload_ps(src_ptr + j, mask);
        __m256 y = _mm256_maskload_ps(ref_ptr + j, mask);

        y = _mm256_add_ps(x, y);

        _mm256_maskstore_ps(ref_ptr + j, mask, y);
    }
}

void insertadd_2(unsigned int length, float* src1_ptr, float* src2_ptr, float* ref_ptr) {
    const unsigned int j = length & ~7u, k = length - j;

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x1 = _mm256_loadu_ps(src1_ptr + i);
        __m256 x2 = _mm256_loadu_ps(src2_ptr + i);
        __m256 y = _mm256_loadu_ps(ref_ptr + i);

        y = _mm256_add_ps(x1, y);
        y = _mm256_add_ps(x2, y);

        _mm256_storeu_ps(ref_ptr + i, y);
    }

    if (k > 0) {
        __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        __m256 x1 = _mm256_maskload_ps(src1_ptr + j, mask);
        __m256 x2 = _mm256_maskload_ps(src2_ptr + j, mask);
        __m256 y = _mm256_maskload_ps(ref_ptr + j, mask);

        y = _mm256_add_ps(x1, y);
        y = _mm256_add_ps(x2, y);

        _mm256_maskstore_ps(ref_ptr + j, mask, y);
    }
}

void insertadd_3(unsigned int length, float* src1_ptr, float* src2_ptr, float* src3_ptr, float* ref_ptr) {
    const unsigned int j = length & ~7u, k = length - j;

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x1 = _mm256_loadu_ps(src1_ptr + i);
        __m256 x2 = _mm256_loadu_ps(src2_ptr + i);
        __m256 x3 = _mm256_loadu_ps(src3_ptr + i);
        __m256 y = _mm256_loadu_ps(ref_ptr + i);

        y = _mm256_add_ps(x1, y);
        y = _mm256_add_ps(x2, y);
        y = _mm256_add_ps(x3, y);

        _mm256_storeu_ps(ref_ptr + i, y);
    }

    if (k > 0) {
        __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        __m256 x1 = _mm256_maskload_ps(src1_ptr + j, mask);
        __m256 x2 = _mm256_maskload_ps(src2_ptr + j, mask);
        __m256 x3 = _mm256_maskload_ps(src3_ptr + j, mask);
        __m256 y = _mm256_maskload_ps(ref_ptr + j, mask);

        y = _mm256_add_ps(x1, y);
        y = _mm256_add_ps(x2, y);
        y = _mm256_add_ps(x3, y);

        _mm256_maskstore_ps(ref_ptr + j, mask, y);
    }
}

void insertadd_4(unsigned int length, float* src1_ptr, float* src2_ptr, float* src3_ptr, float* src4_ptr, float* ref_ptr) {
    const unsigned int j = length & ~7u, k = length - j;

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x1 = _mm256_loadu_ps(src1_ptr + i);
        __m256 x2 = _mm256_loadu_ps(src2_ptr + i);
        __m256 x3 = _mm256_loadu_ps(src3_ptr + i);
        __m256 x4 = _mm256_loadu_ps(src4_ptr + i);
        __m256 y = _mm256_loadu_ps(ref_ptr + i);

        y = _mm256_add_ps(x1, y);
        y = _mm256_add_ps(x2, y);
        y = _mm256_add_ps(x3, y);
        y = _mm256_add_ps(x4, y);

        _mm256_storeu_ps(ref_ptr + i, y);
    }

    if (k > 0) {
        __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        __m256 x1 = _mm256_maskload_ps(src1_ptr + j, mask);
        __m256 x2 = _mm256_maskload_ps(src2_ptr + j, mask);
        __m256 x3 = _mm256_maskload_ps(src3_ptr + j, mask);
        __m256 x4 = _mm256_maskload_ps(src4_ptr + j, mask);
        __m256 y = _mm256_maskload_ps(ref_ptr + j, mask);

        y = _mm256_add_ps(x1, y);
        y = _mm256_add_ps(x2, y);
        y = _mm256_add_ps(x3, y);
        y = _mm256_add_ps(x4, y);

        _mm256_maskstore_ps(ref_ptr + j, mask, y);
    }
}

void TensorShaderAvxBackend::Elementwise::Sum(unsigned int index, unsigned int length, cli::array <cli::array<float>^>^ src, cli::array<float>^ dst) {
    
    Util::CheckOutOfRange(index, length, src);
    Util::CheckOutOfRange(index, length, dst);
    
    if (src->Length == 0) {
        ArrayManipulation::Zeroset(index, length, dst);
        return;
    }
    if (src->Length == 2) {
        Add(index, length, src[0], src[1], dst);
        return;
    }
    
    Array::Copy(src[src->Length - 1], (long)index, dst, index, length);

    int j = (src->Length - 1) & ~3, k = (src->Length - 1) - j;

    for (int i = 0; i < j; i += 4) {
        pin_ptr<float> pinptr_src1 = &src[i][0];
        pin_ptr<float> pinptr_src2 = &src[i + 1][0];
        pin_ptr<float> pinptr_src3 = &src[i + 2][0];
        pin_ptr<float> pinptr_src4 = &src[i + 3][0];
        pin_ptr<float> pinptr_dst = &dst[0];

        float* src1_ptr = pinptr_src1;
        float* src2_ptr = pinptr_src2;
        float* src3_ptr = pinptr_src3;
        float* src4_ptr = pinptr_src4;
        float* dst_ptr = pinptr_dst;

        insertadd_4(length, src1_ptr + index, src2_ptr + index, src3_ptr + index, src4_ptr + index, dst_ptr + index);
    }
    if (k >= 3) {
        pin_ptr<float> pinptr_src1 = &src[j][0];
        pin_ptr<float> pinptr_src2 = &src[j + 1][0];
        pin_ptr<float> pinptr_src3 = &src[j + 2][0];
        pin_ptr<float> pinptr_dst = &dst[0];

        float* src1_ptr = pinptr_src1;
        float* src2_ptr = pinptr_src2;
        float* src3_ptr = pinptr_src3;
        float* dst_ptr = pinptr_dst;

        insertadd_3(length, src1_ptr + index, src2_ptr + index, src3_ptr + index, dst_ptr + index);
    }
    else if (k >= 2) {
        pin_ptr<float> pinptr_src1 = &src[j][0];
        pin_ptr<float> pinptr_src2 = &src[j + 1][0];
        pin_ptr<float> pinptr_dst = &dst[0];

        float* src1_ptr = pinptr_src1;
        float* src2_ptr = pinptr_src2;
        float* dst_ptr = pinptr_dst;

        insertadd_2(length, src1_ptr + index, src2_ptr + index, dst_ptr + index);
    }
    else if (k >= 1) {
        pin_ptr<float> pinptr_src1 = &src[j][0];
        pin_ptr<float> pinptr_dst = &dst[0];

        float* src1_ptr = pinptr_src1;
        float* dst_ptr = pinptr_dst;

        insertadd_1(length, src1_ptr + index, dst_ptr + index);
    }
}
