#include "../../TensorShaderAvxBackend.h"

using namespace System;

void quaternion_purei(unsigned int length, float* srci_ptr, float* dst_ptr) {
    for (unsigned int i = 0, j = 0; i < length; i += 4, j++) {
        dst_ptr[i + 1] = srci_ptr[j];
        dst_ptr[i] = dst_ptr[i + 2] = dst_ptr[i + 3] = 0;
    }
}

void TensorShaderAvxBackend::Quaternion::PureI(unsigned int length, cli::array<float>^ src_i, cli::array<float>^ dst) {

    Util::CheckDuplicateArray(src_i, dst);

    if (length % 4 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckOutOfRange(0, length / 4, src_i);
    Util::CheckOutOfRange(0, length, dst);

    pin_ptr<float> pinptr_srci = &src_i[0];
    pin_ptr<float> pinptr_dst = &dst[0];

    float* srci_ptr = pinptr_srci;
    float* dst_ptr = pinptr_dst;

    quaternion_purei(length, srci_ptr, dst_ptr);
}