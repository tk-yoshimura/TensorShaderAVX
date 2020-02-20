#include "../../TensorShaderAvxBackend.h"

using namespace System;

void quaternion_j(unsigned int length, const float* __restrict src_ptr, float* __restrict dst_ptr) {
    for (unsigned int i = 0, j = 0; i < length; i += 4, j++) {
        dst_ptr[j] = src_ptr[i + 2];
    }
}

void TensorShaderAvxBackend::Quaternion::J(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst_j) {

    Util::CheckDuplicateArray(src, dst_j);

    if (length % 4 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckLength(length, src);
    Util::CheckLength(length / 4, dst_j);

    float* src_ptr = (float*)(src->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst_j->Ptr.ToPointer());

    quaternion_j(length, src_ptr, dst_ptr);
}