#include "../TensorShaderAvxBackend.h"

using namespace System;

void argmin(unsigned int length, unsigned int channels, float* src_ptr, float* dst_ptr) {
    for (unsigned int n = 0; n < length; n++) {
        unsigned int idx = 0;
        float vmin = src_ptr[channels * n];
        
        for (unsigned int i = 1; i < channels; i++) {
            float v = src_ptr[i + channels * n];

            if (v < vmin) {
                idx = i;
                vmin = v;
            }
        }

        dst_ptr[n] = (float)idx;
    }
}

void TensorShaderAvxBackend::Indexer::ArgMin(unsigned int length, unsigned int channels, cli::array<float>^ src, cli::array<float>^ dst) {

    Util::CheckDuplicateArray(src, dst);

    Util::CheckOutOfRange(0, length * channels, src);
    Util::CheckOutOfRange(0, length, dst);
    
    pin_ptr<float> pinptr_src = &src[0];
    pin_ptr<float> pinptr_dst = &dst[0];

    float* src_ptr = pinptr_src;
    float* dst_ptr = pinptr_dst;

    argmin(length, channels, src_ptr, dst_ptr);
}