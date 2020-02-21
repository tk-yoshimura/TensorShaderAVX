#include "../../TensorShaderAvxBackend.h"

using namespace System;

void complex_cast(unsigned int length, 
                  const float* __restrict srcreal_ptr, const float* __restrict srcimag_ptr, float* __restrict dst_ptr) {

    for (unsigned int i = 0, j = 0; i < length; i += 2, j++) {
        dst_ptr[i] = srcreal_ptr[j];
        dst_ptr[i + 1] = srcimag_ptr[j];
    }
}

void TensorShaderAvxBackend::Complex::Cast(unsigned int length, AvxArray<float>^ src_real, AvxArray<float>^ src_imag, AvxArray<float>^ dst) {

    Util::CheckDuplicateArray(src_real, src_imag, dst);

    if (length % 2 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckLength(length / 2, src_real, src_imag);
    Util::CheckLength(length, dst);

    const float* srcreal_ptr = (const float*)(src_real->Ptr.ToPointer());
    const float* srcimag_ptr = (const float*)(src_imag->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    complex_cast(length, srcreal_ptr, srcimag_ptr, dst_ptr);
}