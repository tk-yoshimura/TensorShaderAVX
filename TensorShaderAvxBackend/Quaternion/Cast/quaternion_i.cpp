#include "../../TensorShaderAvxBackend.h"

using namespace System;

void quaternion_i(unsigned int length, float* src_ptr, float* dst_ptr) {
    for (unsigned int i = 0, j = 0; i < length; i += 4, j++) {
        dst_ptr[j] = src_ptr[i + 1];
    }
}

void TensorShaderAvxBackend::Quaternion::I(unsigned int length, cli::array<float>^ src, cli::array<float>^ dst_i) {

    Util::CheckDuplicateArray(src, dst_i);

    if (length % 4 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckOutOfRange(0, length, src);
    Util::CheckOutOfRange(0, length / 4, dst_i);

    pin_ptr<float> pinptr_src = &src[0];
    pin_ptr<float> pinptr_dst = &dst_i[0];

    float* src_ptr = pinptr_src;
    float* dst_ptr = pinptr_dst;

    quaternion_i(length, src_ptr, dst_ptr);
}