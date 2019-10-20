#include "../../TensorShaderAvxBackend.h"

using namespace System;

void quaternion_purer(unsigned int length, float* srcr_ptr, float* dst_ptr) {
    for (unsigned int i = 0, j = 0; i < length; i += 4, j++) {
        dst_ptr[i] = srcr_ptr[j];
        dst_ptr[i + 1] = dst_ptr[i + 2] = dst_ptr[i + 3] = 0;
    }
}

void TensorShaderAvxBackend::Quaternion::PureR(unsigned int length, cli::array<float>^ src_r, cli::array<float>^ dst) {

    Util::CheckDuplicateArray(src_r, dst);

    if (length % 4 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckOutOfRange(0, length / 4, src_r);
    Util::CheckOutOfRange(0, length, dst);

    pin_ptr<float> pinptr_srcr = &src_r[0];
    pin_ptr<float> pinptr_dst = &dst[0];

    float* srcr_ptr = pinptr_srcr;
    float* dst_ptr = pinptr_dst;

    quaternion_purer(length, srcr_ptr, dst_ptr);
}