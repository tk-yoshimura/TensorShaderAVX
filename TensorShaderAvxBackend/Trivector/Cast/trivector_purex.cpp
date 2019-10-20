#include "../../TensorShaderAvxBackend.h"

using namespace System;

void trivector_purex(unsigned int length, float* srcx_ptr, float* dst_ptr) {
    for (unsigned int i = 0, j = 0; i < length; i += 3, j++) {
        dst_ptr[i] = srcx_ptr[j];
        dst_ptr[i + 1] = 0;
        dst_ptr[i + 2] = 0;
    }
}

void TensorShaderAvxBackend::Trivector::PureX(unsigned int length, cli::array<float>^ src_x, cli::array<float>^ dst) {

    Util::CheckDuplicateArray(src_x, dst);

    if (length % 3 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckOutOfRange(0, length / 3, src_x);
    Util::CheckOutOfRange(0, length, dst);

    pin_ptr<float> pinptr_srcx = &src_x[0];
    pin_ptr<float> pinptr_dst = &dst[0];

    float* srcx_ptr = pinptr_srcx;
    float* dst_ptr = pinptr_dst;

    trivector_purex(length, srcx_ptr, dst_ptr);
}