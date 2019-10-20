#include "../../TensorShaderAvxBackend.h"

using namespace System;

void trivector_cast(unsigned int length, float* srcx_ptr, float* srcy_ptr, float* srcz_ptr, float* dst_ptr) {
    for (unsigned int i = 0, j = 0; i < length; i += 3, j++) {
        dst_ptr[i] = srcx_ptr[j];
        dst_ptr[i + 1] = srcy_ptr[j];
        dst_ptr[i + 2] = srcz_ptr[j];
    }
}

void TensorShaderAvxBackend::Trivector::Cast(unsigned int length, cli::array<float>^ src_x, cli::array<float>^ src_y, cli::array<float>^ src_z, cli::array<float>^ dst) {

    Util::CheckDuplicateArray(src_x, src_y, src_z, dst);

    if (length % 3 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckOutOfRange(0, length / 3, src_x, src_y, src_z);
    Util::CheckOutOfRange(0, length, dst);

    pin_ptr<float> pinptr_srcx = &src_x[0];
    pin_ptr<float> pinptr_srcy = &src_y[0];
    pin_ptr<float> pinptr_srcz = &src_z[0];
    pin_ptr<float> pinptr_dst = &dst[0];

    float* srcx_ptr = pinptr_srcx;
    float* srcy_ptr = pinptr_srcy;
    float* srcz_ptr = pinptr_srcz;
    float* dst_ptr = pinptr_dst;

    trivector_cast(length, srcx_ptr, srcy_ptr, srcz_ptr, dst_ptr);
}