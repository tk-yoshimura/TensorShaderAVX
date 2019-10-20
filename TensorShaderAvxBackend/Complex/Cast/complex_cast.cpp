#include "../../TensorShaderAvxBackend.h"

using namespace System;

void complex_cast(unsigned int length, float* srcreal_ptr, float* srcimag_ptr, float* dst_ptr) {
    for (unsigned int i = 0, j = 0; i < length; i += 2, j++) {
        dst_ptr[i] = srcreal_ptr[j];
        dst_ptr[i + 1] = srcimag_ptr[j];
    }
}

void TensorShaderAvxBackend::Complex::Cast(unsigned int length, cli::array<float>^ src_real, cli::array<float>^ src_imag, cli::array<float>^ dst) {

    Util::CheckDuplicateArray(src_real, src_imag, dst);

    if (length % 2 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckOutOfRange(0, length / 2, src_real, src_imag);
    Util::CheckOutOfRange(0, length, dst);

    pin_ptr<float> pinptr_srcreal = &src_real[0];
    pin_ptr<float> pinptr_srcimag = &src_imag[0];
    pin_ptr<float> pinptr_dst = &dst[0];

    float* srcreal_ptr = pinptr_srcreal;
    float* srcimag_ptr = pinptr_srcimag;
    float* dst_ptr = pinptr_dst;

    complex_cast(length, srcreal_ptr, srcimag_ptr, dst_ptr);
}