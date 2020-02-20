#include "../TensorShaderAvxBackend.h"

using namespace System;

void argmax(unsigned int length, unsigned int channels, const float* __restrict src_ptr, float* __restrict dst_ptr) {
    for (unsigned int n = 0; n < length; n++) {
        unsigned int idx = 0;
        float vmax = src_ptr[channels * n];
        
        for (unsigned int i = 1; i < channels; i++) {
            float v = src_ptr[i + channels * n];

            if (v > vmax) {
                idx = i;
                vmax = v;
            }
        }

        dst_ptr[n] = (float)idx;
    }
}

void TensorShaderAvxBackend::Indexer::ArgMax(unsigned int length, unsigned int channels, AvxArray<float>^ src, AvxArray<float>^ dst) {

    Util::CheckDuplicateArray(src, dst);

    Util::CheckLength(length * channels, src);
    Util::CheckLength(length, dst);
    
    const float* src_ptr = (const float*)(src->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    argmax(length, channels, src_ptr, dst_ptr);
}