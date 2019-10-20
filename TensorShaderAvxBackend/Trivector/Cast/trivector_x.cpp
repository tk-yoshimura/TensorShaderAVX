#include "../../TensorShaderAvxBackend.h"

using namespace System;

void trivector_x(unsigned int length, float* src_ptr, float* dst_ptr) {
    for (unsigned int i = 0, j = 0; i < length; i += 3, j++) {
        dst_ptr[j] = src_ptr[i];
    }
}

void TensorShaderAvxBackend::Trivector::X(unsigned int length, cli::array<float>^ src, cli::array<float>^ dst_x) {

    Util::CheckDuplicateArray(src, dst_x);

    if (length % 3 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckOutOfRange(0, length, src);
    Util::CheckOutOfRange(0, length / 3, dst_x);

    pin_ptr<float> pinptr_src = &src[0];
    pin_ptr<float> pinptr_dst = &dst_x[0];

    float* src_ptr = pinptr_src;
    float* dst_ptr = pinptr_dst;

    trivector_x(length, src_ptr, dst_ptr);
}