#include "../../TensorShaderAvxBackend.h"

using namespace System;

void complex_real(unsigned int length, float* src_ptr, float* dst_ptr) {
    for (unsigned int i = 0, j = 0; i < length; i += 2, j++) {
        dst_ptr[j] = src_ptr[i];
    }
}

void TensorShaderAvxBackend::Complex::Real(unsigned int length, cli::array<float>^ src, cli::array<float>^ dst_real) {

    Util::CheckDuplicateArray(src, dst_real);

    if (length % 2 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckOutOfRange(0, length, src);
    Util::CheckOutOfRange(0, length / 2, dst_real);

    pin_ptr<float> pinptr_src = &src[0];
    pin_ptr<float> pinptr_dst = &dst_real[0];

    float* src_ptr = pinptr_src;
    float* dst_ptr = pinptr_dst;

    complex_real(length, src_ptr, dst_ptr);
}