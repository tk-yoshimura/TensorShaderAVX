#include "../TensorShaderAvxBackend.h"

using namespace System;

void mul_bnwise(unsigned int veclength, unsigned int maplength, float* srcvec_ptr, float* srcmap_ptr, float* dstmap_ptr) {
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

void TensorShaderAvxBackend::Batchwise::Mul(unsigned int vector_length, unsigned int map_length, cli::array<float>^ srcvector, cli::array<float>^ srcmap, cli::array<float>^ dstmap) {

    Util::CheckDuplicateArray(srcvector, srcmap, dstmap);

    Util::CheckOutOfRange(0, vector_length, srcvector);
    Util::CheckOutOfRange(0, map_length, srcmap);
    Util::CheckOutOfRange(0, map_length, dstmap);

    if (vector_length == 0 || (map_length % vector_length) != 0) {
        throw gcnew System::ArgumentException();
    }

    if (vector_length == 1) {
        Elementwise::MulConstant(0, map_length, srcvector[0], srcmap, dstmap);
        return;
    }

    pin_ptr<float> pinptr_srcvec = &srcvector[0];
    pin_ptr<float> pinptr_srcmap = &srcmap[0];
    pin_ptr<float> pinptr_dstmap = &dstmap[0];

    float* srcvec_ptr = pinptr_srcvec;
    float* srcmap_ptr = pinptr_srcmap;
    float* dstmap_ptr = pinptr_dstmap;

    mul_bnwise(vector_length, map_length, srcvec_ptr, srcmap_ptr, dstmap_ptr);
}
