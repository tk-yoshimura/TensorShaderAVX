#include "../../TensorShaderAvxBackend.h"

using namespace System;

void quaternion_r(unsigned int length, float* src_ptr, float* dst_ptr) {
    for (unsigned int i = 0, j = 0; i < length; i += 4, j++) {
        dst_ptr[j] = src_ptr[i];
    }
}

void TensorShaderAvxBackend::Quaternion::R(unsigned int length, cli::array<float>^ src, cli::array<float>^ dst_r) {

    Util::CheckDuplicateArray(src, dst_r);

    if (length % 4 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckOutOfRange(0, length, src);
    Util::CheckOutOfRange(0, length / 4, dst_r);

    pin_ptr<float> pinptr_src = &src[0];
    pin_ptr<float> pinptr_dst = &dst_r[0];

    float* src_ptr = pinptr_src;
    float* dst_ptr = pinptr_dst;

    quaternion_r(length, src_ptr, dst_ptr);
}