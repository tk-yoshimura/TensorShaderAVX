#include "../../TensorShaderAvxBackend.h"

using namespace System;

void complex_pureimag(unsigned int length, const float* __restrict srcimag_ptr, float* __restrict dst_ptr) {
    for (unsigned int i = 0, j = 0; i < length; i += 2, j++) {
        dst_ptr[i] = 0;
        dst_ptr[i + 1] = srcimag_ptr[j];
    }
}

void TensorShaderAvxBackend::Complex::PureImag(unsigned int length, AvxArray<float>^ src_imag, AvxArray<float>^ dst) {

    Util::CheckDuplicateArray(src_imag, dst);

    if (length % 2 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckLength(length / 2, src_imag);
    Util::CheckLength(length, dst);

    const float* srcimag_ptr = (const float*)(src_imag->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    complex_pureimag(length, srcimag_ptr, dst_ptr);
}