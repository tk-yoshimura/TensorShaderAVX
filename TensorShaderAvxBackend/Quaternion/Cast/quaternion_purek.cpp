#include "../../TensorShaderAvxBackend.h"

using namespace System;

void quaternion_purek(unsigned int length, float* srck_ptr, float* dst_ptr) {
    for (unsigned int i = 0, j = 0; i < length; i += 4, j++) {
        dst_ptr[i + 3] = srck_ptr[j];
        dst_ptr[i + 0] = dst_ptr[i + 1] = dst_ptr[i + 2] = 0;
    }
}

void TensorShaderAvxBackend::Quaternion::PureK(unsigned int length, cli::array<float>^ src_k, cli::array<float>^ dst) {

    Util::CheckDuplicateArray(src_k, dst);

    if (length % 4 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckOutOfRange(0, length / 4, src_k);
    Util::CheckOutOfRange(0, length, dst);

    pin_ptr<float> pinptr_srck = &src_k[0];
    pin_ptr<float> pinptr_dst = &dst[0];

    float* srck_ptr = pinptr_srck;
    float* dst_ptr = pinptr_dst;

    quaternion_purek(length, srck_ptr, dst_ptr);
}