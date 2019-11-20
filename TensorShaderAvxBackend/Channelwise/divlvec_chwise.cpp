#include "../TensorShaderAvxBackend.h"

using namespace System;

void divlvec_chwise(unsigned int veclength, unsigned int maplength, float* srcvec_ptr, float* srcmap_ptr, float* dstmap_ptr) {
    const unsigned int maplength_sep = maplength / veclength * veclength, maplength_rem = maplength - maplength_sep;

    {
        const unsigned int j = veclength & ~7, k = veclength - j;
        const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        for (unsigned int mapindex = 0; mapindex < maplength_sep; mapindex += veclength) {
            for (unsigned int i = 0; i < j; i += 8) {
                __m256 x1 = _mm256_loadu_ps(srcvec_ptr + i);
                __m256 x2 = _mm256_loadu_ps(srcmap_ptr + mapindex + i);

                __m256 y = _mm256_div_ps(x1, x2);

                _mm256_storeu_ps(dstmap_ptr + mapindex + i, y);
            }

            if (k > 0) {
                __m256 x1 = _mm256_maskload_ps(srcvec_ptr + j, mask);
                __m256 x2 = _mm256_maskload_ps(srcmap_ptr + mapindex + j, mask);

                __m256 y = _mm256_div_ps(x1, x2);

                _mm256_maskstore_ps(dstmap_ptr + mapindex + j, mask, y);
            }
        }
    }

    if (maplength_rem > 0) {
        const unsigned int j = maplength_rem & ~7, k = maplength_rem - j;
        const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        for (unsigned int i = 0; i < j; i += 8) {
            __m256 x1 = _mm256_loadu_ps(srcvec_ptr + i);
            __m256 x2 = _mm256_loadu_ps(srcmap_ptr + maplength_sep + i);

            __m256 y = _mm256_div_ps(x1, x2);

            _mm256_storeu_ps(dstmap_ptr + maplength_sep + i, y);
        }

        if (k > 0) {
            __m256 x1 = _mm256_maskload_ps(srcvec_ptr + j, mask);
            __m256 x2 = _mm256_maskload_ps(srcmap_ptr + maplength_sep + j, mask);

            __m256 y = _mm256_div_ps(x1, x2);

            _mm256_maskstore_ps(dstmap_ptr + maplength_sep + j, mask, y);
        }
    }
}

void TensorShaderAvxBackend::Channelwise::DivLVector(unsigned int vector_length, unsigned int map_length, AvxArray<float>^ srcvector, AvxArray<float>^ srcmap, AvxArray<float>^ dstmap) {

    Util::CheckDuplicateArray(srcvector, srcmap, dstmap);

    Util::CheckLength(vector_length, srcvector);
    Util::CheckLength(map_length, srcmap, dstmap);
    
    while (vector_length < 64 && vector_length % 8 != 0) {
        AvxArray<float>^ v = gcnew AvxArray<float>(vector_length * 2);

        AvxArray<float>::Copy(srcvector, v, (int)vector_length);
        AvxArray<float>::Copy(srcvector, 0, v, vector_length, vector_length);

        srcvector = v;
        vector_length *= 2;
    }

    float* srcvec_ptr = (float*)(srcvector->Ptr.ToPointer());
    float* srcmap_ptr = (float*)(srcmap->Ptr.ToPointer());
    float* dstmap_ptr = (float*)(dstmap->Ptr.ToPointer());
    
    if (vector_length == 1) {
        Elementwise::DivConstant(map_length, srcvec_ptr[0], srcmap, dstmap);
        return;
    }

    divlvec_chwise(vector_length, map_length, srcvec_ptr, srcmap_ptr, dstmap_ptr);
}
