#include "../TensorShaderAvxBackend.h"

using namespace System;

void flip(unsigned int stride, unsigned int axislength, unsigned int slides, float* src_ptr, float* dst_ptr) {
    for (unsigned int k = 0, src_idx = 0; k < slides; k++) {
        for (unsigned int j = 0; j < axislength; j++) {
            for (unsigned int i = 0, dst_idx = stride * ((axislength - j - 1) + axislength * k); i < stride; i++, src_idx++, dst_idx++) {
                dst_ptr[dst_idx] = src_ptr[src_idx];
            }
        }
    }
}

void TensorShaderAvxBackend::ArrayManipulation::Flip(unsigned int stride, unsigned int axislength, unsigned int slides, cli::array<float>^ src, cli::array<float>^ dst) {

    Util::CheckDuplicateArray(src, dst);
    Util::CheckOutOfRange(0, stride * axislength * slides, src, dst);

    pin_ptr<float> pinptr_src = &src[0];
    pin_ptr<float> pinptr_dst = &dst[0];

    float* src_ptr = pinptr_src;
    float* dst_ptr = pinptr_dst;

    flip(stride, axislength, slides, src_ptr, dst_ptr);
}