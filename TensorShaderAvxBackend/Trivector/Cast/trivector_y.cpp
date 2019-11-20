#include "../../TensorShaderAvxBackend.h"

using namespace System;

void trivector_y(unsigned int length, float* src_ptr, float* dst_ptr) {
    for (unsigned int i = 0, j = 0; i < length; i += 3, j++) {
        dst_ptr[j] = src_ptr[i + 1];
    }
}

void TensorShaderAvxBackend::Trivector::Y(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst_y) {

    Util::CheckDuplicateArray(src, dst_y);

    if (length % 3 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckLength(length, src);
    Util::CheckLength(length / 3, dst_y);

    float* src_ptr = (float*)(src->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst_y->Ptr.ToPointer());

    trivector_y(length, src_ptr, dst_ptr);
}