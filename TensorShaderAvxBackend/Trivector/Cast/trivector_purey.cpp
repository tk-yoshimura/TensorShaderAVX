#include "../../TensorShaderAvxBackend.h"

using namespace System;

void trivector_purey(unsigned int length, const float* __restrict srcy_ptr, float* __restrict dst_ptr) {

    for (unsigned int i = 0, j = 0; i < length; i += 3, j++) {
        dst_ptr[i] = 0;
        dst_ptr[i + 1] = srcy_ptr[j];
        dst_ptr[i + 2] = 0;
    }
}

void TensorShaderAvxBackend::Trivector::PureY(unsigned int length, AvxArray<float>^ src_y, AvxArray<float>^ dst) {

    Util::CheckDuplicateArray(src_y, dst);

    if (length % 3 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckLength(length / 3, src_y);
    Util::CheckLength(length, dst);

    const float* srcy_ptr = (const float*)(src_y->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    trivector_purey(length, srcy_ptr, dst_ptr);
}