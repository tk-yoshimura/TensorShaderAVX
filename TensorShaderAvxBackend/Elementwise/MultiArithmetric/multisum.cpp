#include "../../TensorShaderAvxBackend.h"

using namespace System;

void insertadd_1(unsigned int length, const float* __restrict src_ptr, float* __restrict ref_ptr) {
    const unsigned int j = length & ~7u, k = length - j;

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x = _mm256_load_ps(src_ptr + i);
        __m256 y = _mm256_load_ps(ref_ptr + i);

        y = _mm256_add_ps(x, y);

        _mm256_store_ps(ref_ptr + i, y);
    }

    if (k > 0) {
        __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        __m256 x = _mm256_maskload_ps(src_ptr + j, mask);
        __m256 y = _mm256_maskload_ps(ref_ptr + j, mask);

        y = _mm256_add_ps(x, y);

        _mm256_maskstore_ps(ref_ptr + j, mask, y);
    }
}

void insertadd_2(unsigned int length, const float* __restrict src1_ptr, const float* __restrict src2_ptr, float* __restrict ref_ptr) {
    const unsigned int j = length & ~7u, k = length - j;

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x1 = _mm256_load_ps(src1_ptr + i);
        __m256 x2 = _mm256_load_ps(src2_ptr + i);
        __m256 y = _mm256_load_ps(ref_ptr + i);

        y = _mm256_add_ps(x1, y);
        y = _mm256_add_ps(x2, y);

        _mm256_store_ps(ref_ptr + i, y);
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

void insertadd_3(unsigned int length, const float* __restrict src1_ptr, const float* __restrict src2_ptr, const float* __restrict src3_ptr, float* __restrict ref_ptr) {
    const unsigned int j = length & ~7u, k = length - j;

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x1 = _mm256_load_ps(src1_ptr + i);
        __m256 x2 = _mm256_load_ps(src2_ptr + i);
        __m256 x3 = _mm256_load_ps(src3_ptr + i);
        __m256 y = _mm256_load_ps(ref_ptr + i);

        y = _mm256_add_ps(x1, y);
        y = _mm256_add_ps(x2, y);
        y = _mm256_add_ps(x3, y);

        _mm256_store_ps(ref_ptr + i, y);
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

void insertadd_4(unsigned int length, const float* __restrict src1_ptr, const float* __restrict src2_ptr, const float* __restrict src3_ptr, const float* __restrict src4_ptr, float* __restrict ref_ptr) {
    const unsigned int j = length & ~7u, k = length - j;

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x1 = _mm256_load_ps(src1_ptr + i);
        __m256 x2 = _mm256_load_ps(src2_ptr + i);
        __m256 x3 = _mm256_load_ps(src3_ptr + i);
        __m256 x4 = _mm256_load_ps(src4_ptr + i);
        __m256 y = _mm256_load_ps(ref_ptr + i);

        y = _mm256_add_ps(x1, y);
        y = _mm256_add_ps(x2, y);
        y = _mm256_add_ps(x3, y);
        y = _mm256_add_ps(x4, y);

        _mm256_store_ps(ref_ptr + i, y);
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

void TensorShaderAvxBackend::Elementwise::Sum(unsigned int length, cli::array <AvxArray<float>^>^ src, AvxArray<float>^ dst) {
    
    Util::CheckLength(length, src);
    Util::CheckLength(length, dst);
    
    if (src->Length == 0) {
        dst->Zeroset();
        return;
    }
    if (src->Length == 2) {
        Add(length, src[0], src[1], dst);
        return;
    }
    
    AvxArray<float>::Copy(src[src->Length - 1], dst, length);

    int j = (src->Length - 1) & ~3, k = (src->Length - 1) - j;

    for (int i = 0; i < j; i += 4) {
        const float* src1_ptr = (const float*)(src[i]->Ptr.ToPointer());
        const float* src2_ptr = (const float*)(src[i + 1]->Ptr.ToPointer());
        const float* src3_ptr = (const float*)(src[i + 2]->Ptr.ToPointer());
        const float* src4_ptr = (const float*)(src[i + 3]->Ptr.ToPointer());
        float* dst_ptr = (float*)(dst->Ptr.ToPointer());

        insertadd_4(length, src1_ptr, src2_ptr, src3_ptr, src4_ptr, dst_ptr);
    }
    if (k >= 3) {
        const float* src1_ptr = (const float*)(src[j]->Ptr.ToPointer());
        const float* src2_ptr = (const float*)(src[j + 1]->Ptr.ToPointer());
        const float* src3_ptr = (const float*)(src[j + 2]->Ptr.ToPointer());
        float* dst_ptr = (float*)(dst->Ptr.ToPointer());

        insertadd_3(length, src1_ptr, src2_ptr, src3_ptr, dst_ptr);
    }
    else if (k >= 2) {
        const float* src1_ptr = (const float*)(src[j]->Ptr.ToPointer());
        const float* src2_ptr = (const float*)(src[j + 1]->Ptr.ToPointer());
        float* dst_ptr = (float*)(dst->Ptr.ToPointer());

        insertadd_2(length, src1_ptr, src2_ptr, dst_ptr);
    }
    else if (k >= 1) {
        const float* src1_ptr = (const float*)(src[j]->Ptr.ToPointer());
        float* dst_ptr = (float*)(dst->Ptr.ToPointer());

        insertadd_1(length, src1_ptr, dst_ptr);
    }
}
