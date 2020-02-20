#include "../TensorShaderAvxBackend.h"

using namespace System;

void mul_bnwise(unsigned int veclength, unsigned int maplength, 
                const float* __restrict srcvec_ptr, const float* __restrict srcmap_ptr, float* __restrict dstmap_ptr) {

    const unsigned int length = maplength / veclength;
    const unsigned int j = length & ~7u, k = length - j;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

    for (unsigned int v = 0; v < veclength; v++) {
        const __m256 xv = _mm256_set1_ps(srcvec_ptr[v]);

        for (unsigned int i = 0; i < j; i += 8) {
            __m256 x = _mm256_loadu_ps(srcmap_ptr + i);

            __m256 y = _mm256_mul_ps(x, xv);

            _mm256_storeu_ps(dstmap_ptr + i, y);
        }

        if (k > 0) {
            __m256 x = _mm256_maskload_ps(srcmap_ptr + j, mask);

            __m256 y = _mm256_mul_ps(x, xv);

            _mm256_maskstore_ps(dstmap_ptr + j, mask, y);
        }

        srcmap_ptr += length;
        dstmap_ptr += length;
    }
}

void TensorShaderAvxBackend::Batchwise::Mul(unsigned int vector_length, unsigned int map_length, AvxArray<float>^ srcvector, AvxArray<float>^ srcmap, AvxArray<float>^ dstmap) {

    Util::CheckDuplicateArray(srcvector, srcmap, dstmap);

    if (vector_length == 0 || (map_length % vector_length) != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckLength(vector_length, srcvector);
    Util::CheckLength(map_length, srcmap, dstmap);

    const float* srcvec_ptr = (const float*)(srcvector->Ptr.ToPointer());
    const float* srcmap_ptr = (const float*)(srcmap->Ptr.ToPointer());
    float* dstmap_ptr = (float*)(dstmap->Ptr.ToPointer());

    if (vector_length == 1) {
        Elementwise::MulConstant(map_length, srcvec_ptr[0], srcmap, dstmap);
        return;
    }

    mul_bnwise(vector_length, map_length, srcvec_ptr, srcmap_ptr, dstmap_ptr);
}
