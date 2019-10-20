#include "../../TensorShaderAvxBackend.h"

using namespace System;

void quaternion_j(unsigned int length, float* src_ptr, float* dst_ptr) {
    for (unsigned int i = 0, j = 0; i < length; i += 4, j++) {
        dst_ptr[j] = src_ptr[i + 2];
    }
}

void TensorShaderAvxBackend::Quaternion::J(unsigned int length, cli::array<float>^ src, cli::array<float>^ dst_j) {

    Util::CheckDuplicateArray(src, dst_j);

    if (length % 4 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckOutOfRange(0, length, src);
    Util::CheckOutOfRange(0, length / 4, dst_j);

    pin_ptr<float> pinptr_src = &src[0];
    pin_ptr<float> pinptr_dst = &dst_j[0];

    float* src_ptr = pinptr_src;
    float* dst_ptr = pinptr_dst;

    quaternion_j(length, src_ptr, dst_ptr);
}