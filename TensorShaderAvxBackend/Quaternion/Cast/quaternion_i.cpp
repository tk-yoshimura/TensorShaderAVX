#include "../../TensorShaderAvxBackend.h"

using namespace System;

void quaternion_i(unsigned int length, const float* __restrict src_ptr, float* __restrict dst_ptr) {
    for (unsigned int i = 0, j = 0; i < length; i += 4, j++) {
        dst_ptr[j] = src_ptr[i + 1];
    }
}

void TensorShaderAvxBackend::Quaternion::I(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst_i) {

    Util::CheckDuplicateArray(src, dst_i);

    if (length % 4 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckLength(length, src);
    Util::CheckLength(length / 4, dst_i);

    float* src_ptr = (float*)(src->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst_i->Ptr.ToPointer());

    quaternion_i(length, src_ptr, dst_ptr);
}