#include "../../TensorShaderAvxBackend.h"

using namespace System;

void complex_purereal(unsigned int length, float* srcreal_ptr, float* dst_ptr) {
    for (unsigned int i = 0, j = 0; i < length; i += 2, j++) {
        dst_ptr[i] = srcreal_ptr[j];
        dst_ptr[i + 1] = 0;
    }
}

void TensorShaderAvxBackend::Complex::PureReal(unsigned int length, AvxArray<float>^ src_real, AvxArray<float>^ dst) {

    Util::CheckDuplicateArray(src_real, dst);

    if (length % 2 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckLength(length / 2, src_real);
    Util::CheckLength(length, dst);

    float* srcreal_ptr = (float*)(src_real->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    complex_purereal(length, srcreal_ptr, dst_ptr);
}