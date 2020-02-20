#include "../../TensorShaderAvxBackend.h"

using namespace System;

void complex_real(unsigned int length, const float* __restrict src_ptr, float* __restrict dst_ptr) {
    for (unsigned int i = 0, j = 0; i < length; i += 2, j++) {
        dst_ptr[j] = src_ptr[i];
    }
}

void TensorShaderAvxBackend::Complex::Real(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst_real) {

    Util::CheckDuplicateArray(src, dst_real);

    if (length % 2 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckLength(length, src);
    Util::CheckLength(length / 2, dst_real);

    const float* src_ptr = (const float*)(src->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst_real->Ptr.ToPointer());

    complex_real(length, src_ptr, dst_ptr);
}