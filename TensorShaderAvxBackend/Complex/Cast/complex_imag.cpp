#include "../../TensorShaderAvxBackend.h"

using namespace System;

void complex_imag(unsigned int length, const float* __restrict src_ptr, float* __restrict dst_ptr) {
    for (unsigned int i = 0, j = 0; i < length; i += 2, j++) {
        dst_ptr[j] = src_ptr[i + 1];
    }
}

void TensorShaderAvxBackend::Complex::Imag(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst_imag) {

    Util::CheckDuplicateArray(src, dst_imag);

    if (length % 2 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckLength(length, src);
    Util::CheckLength(length / 2, dst_imag);
    
    const float* src_ptr = (const float*)(src->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst_imag->Ptr.ToPointer());

    complex_imag(length, src_ptr, dst_ptr);
}