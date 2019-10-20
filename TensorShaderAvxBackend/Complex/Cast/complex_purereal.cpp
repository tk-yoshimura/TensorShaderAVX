#include "../../TensorShaderAvxBackend.h"

using namespace System;

void complex_purereal(unsigned int length, float* srcreal_ptr, float* dst_ptr) {
    for (unsigned int i = 0, j = 0; i < length; i += 2, j++) {
        dst_ptr[i] = srcreal_ptr[j];
        dst_ptr[i + 1] = 0;
    }
}

void TensorShaderAvxBackend::Complex::PureReal(unsigned int length, cli::array<float>^ src_real, cli::array<float>^ dst) {

    Util::CheckDuplicateArray(src_real, dst);

    if (length % 2 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckOutOfRange(0, length / 2, src_real);
    Util::CheckOutOfRange(0, length, dst);

    pin_ptr<float> pinptr_srcreal = &src_real[0];
    pin_ptr<float> pinptr_dst = &dst[0];

    float* srcreal_ptr = pinptr_srcreal;
    float* dst_ptr = pinptr_dst;

    complex_purereal(length, srcreal_ptr, dst_ptr);
}