#include "../../TensorShaderAvxBackend.h"

using namespace System;

void complex_pureimag(unsigned int length, float* srcimag_ptr, float* dst_ptr) {
    for (unsigned int i = 0, j = 0; i < length; i += 2, j++) {
        dst_ptr[i] = 0;
        dst_ptr[i + 1] = srcimag_ptr[j];
    }
}

void TensorShaderAvxBackend::Complex::PureImag(unsigned int length, cli::array<float>^ src_imag, cli::array<float>^ dst) {

    Util::CheckDuplicateArray(src_imag, dst);

    if (length % 2 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckOutOfRange(0, length / 2, src_imag);
    Util::CheckOutOfRange(0, length, dst);

    pin_ptr<float> pinptr_srcimag = &src_imag[0];
    pin_ptr<float> pinptr_dst = &dst[0];

    float* srcimag_ptr = pinptr_srcimag;
    float* dst_ptr = pinptr_dst;

    complex_pureimag(length, srcimag_ptr, dst_ptr);
}