#include "../../TensorShaderAvxBackend.h"

using namespace System;

void trivector_purez(unsigned int length, float* srcz_ptr, float* dst_ptr) {
    for (unsigned int i = 0, j = 0; i < length; i += 3, j++) {
        dst_ptr[i] = 0;
        dst_ptr[i + 1] = 0;
        dst_ptr[i + 2] = srcz_ptr[j];
    }
}

void TensorShaderAvxBackend::Trivector::PureZ(unsigned int length, cli::array<float>^ src_z, cli::array<float>^ dst) {

    Util::CheckDuplicateArray(src_z, dst);

    if (length % 3 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckOutOfRange(0, length / 3, src_z);
    Util::CheckOutOfRange(0, length, dst);

    pin_ptr<float> pinptr_srcz = &src_z[0];
    pin_ptr<float> pinptr_dst = &dst[0];

    float* srcz_ptr = pinptr_srcz;
    float* dst_ptr = pinptr_dst;

    trivector_purez(length, srcz_ptr, dst_ptr);
}