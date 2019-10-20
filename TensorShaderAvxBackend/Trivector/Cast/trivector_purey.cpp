#include "../../TensorShaderAvxBackend.h"

using namespace System;

void trivector_purey(unsigned int length, float* srcy_ptr, float* dst_ptr) {
    for (unsigned int i = 0, j = 0; i < length; i += 3, j++) {
        dst_ptr[i] = 0;
        dst_ptr[i + 1] = srcy_ptr[j];
        dst_ptr[i + 2] = 0;
    }
}

void TensorShaderAvxBackend::Trivector::PureY(unsigned int length, cli::array<float>^ src_y, cli::array<float>^ dst) {

    Util::CheckDuplicateArray(src_y, dst);

    if (length % 3 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckOutOfRange(0, length / 3, src_y);
    Util::CheckOutOfRange(0, length, dst);

    pin_ptr<float> pinptr_srcy = &src_y[0];
    pin_ptr<float> pinptr_dst = &dst[0];

    float* srcy_ptr = pinptr_srcy;
    float* dst_ptr = pinptr_dst;

    trivector_purey(length, srcy_ptr, dst_ptr);
}