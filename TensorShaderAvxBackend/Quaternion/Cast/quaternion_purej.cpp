#include "../../TensorShaderAvxBackend.h"

using namespace System;

void quaternion_purej(unsigned int length, float* srcj_ptr, float* dst_ptr) {
    for (unsigned int i = 0, j = 0; i < length; i += 4, j++) {
        dst_ptr[i + 2] = srcj_ptr[j];
        dst_ptr[i] = dst_ptr[i + 1] = dst_ptr[i + 3] = 0;
    }
}

void TensorShaderAvxBackend::Quaternion::PureJ(unsigned int length, cli::array<float>^ src_j, cli::array<float>^ dst) {

    Util::CheckDuplicateArray(src_j, dst);

    if (length % 4 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckOutOfRange(0, length / 4, src_j);
    Util::CheckOutOfRange(0, length, dst);

    pin_ptr<float> pinptr_srcj = &src_j[0];
    pin_ptr<float> pinptr_dst = &dst[0];

    float* srcj_ptr = pinptr_srcj;
    float* dst_ptr = pinptr_dst;

    quaternion_purej(length, srcj_ptr, dst_ptr);
}