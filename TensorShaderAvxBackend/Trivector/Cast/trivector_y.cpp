#include "../../TensorShaderAvxBackend.h"

using namespace System;

void trivector_y(unsigned int length, float* src_ptr, float* dst_ptr) {
    for (unsigned int i = 0, j = 0; i < length; i += 3, j++) {
        dst_ptr[j] = src_ptr[i + 1];
    }
}

void TensorShaderAvxBackend::Trivector::Y(unsigned int length, cli::array<float>^ src, cli::array<float>^ dst_y) {

    Util::CheckDuplicateArray(src, dst_y);

    if (length % 3 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckOutOfRange(0, length, src);
    Util::CheckOutOfRange(0, length / 3, dst_y);

    pin_ptr<float> pinptr_src = &src[0];
    pin_ptr<float> pinptr_dst = &dst_y[0];

    float* src_ptr = pinptr_src;
    float* dst_ptr = pinptr_dst;

    trivector_y(length, src_ptr, dst_ptr);
}