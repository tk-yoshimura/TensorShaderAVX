#include "../TensorShaderAvxBackend.h"

using namespace System;

void flip(unsigned int stride, unsigned int axislength, unsigned int slides, 
          const float* __restrict src_ptr, float* __restrict dst_ptr) {

    for (unsigned int k = 0, src_idx = 0; k < slides; k++) {
        for (unsigned int j = 0; j < axislength; j++) {
            for (unsigned int i = 0, dst_idx = stride * ((axislength - j - 1) + axislength * k); i < stride; i++, src_idx++, dst_idx++) {
                dst_ptr[dst_idx] = src_ptr[src_idx];
            }
        }
    }
}

void TensorShaderAvxBackend::ArrayManipulation::Flip(unsigned int stride, unsigned int axislength, unsigned int slides, AvxArray<float>^ src, AvxArray<float>^ dst) {

    Util::CheckDuplicateArray(src, dst);
    Util::CheckLength(stride * axislength * slides, src, dst);

    const float* src_ptr = (const float*)(src->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    flip(stride, axislength, slides, src_ptr, dst_ptr);
}