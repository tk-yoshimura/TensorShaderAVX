#include "../../TensorShaderAvxBackend.h"

using namespace System;

void trivector_x(unsigned int length, const float* __restrict src_ptr, float* __restrict dst_ptr) {
    for (unsigned int i = 0, j = 0; i < length; i += 3, j++) {
        dst_ptr[j] = src_ptr[i];
    }
}

void TensorShaderAvxBackend::Trivector::X(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst_x) {

    Util::CheckDuplicateArray(src, dst_x);

    if (length % 3 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckLength(length, src);
    Util::CheckLength(length / 3, dst_x);

    const float* src_ptr = (const float*)(src->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst_x->Ptr.ToPointer());

    trivector_x(length, src_ptr, dst_ptr);
}