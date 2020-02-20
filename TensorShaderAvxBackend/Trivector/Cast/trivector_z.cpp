#include "../../TensorShaderAvxBackend.h"

using namespace System;

void trivector_z(unsigned int length, const float* __restrict src_ptr, float* __restrict dst_ptr) {
    for (unsigned int i = 0, j = 0; i < length; i += 3, j++) {
        dst_ptr[j] = src_ptr[i + 2];
    }
}

void TensorShaderAvxBackend::Trivector::Z(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst_z) {

    Util::CheckDuplicateArray(src, dst_z);

    if (length % 3 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckLength(length, src);
    Util::CheckLength(length / 3, dst_z);

    float* src_ptr = (float*)(src->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst_z->Ptr.ToPointer());

    trivector_z(length, src_ptr, dst_ptr);
}