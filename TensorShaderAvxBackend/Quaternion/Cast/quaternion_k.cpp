#include "../../TensorShaderAvxBackend.h"

using namespace System;

void quaternion_k(unsigned int length, const float* __restrict src_ptr, float* __restrict dst_ptr) {
    for (unsigned int i = 0, j = 0; i < length; i += 4, j++) {
        dst_ptr[j] = src_ptr[i + 3];
    }
}

void TensorShaderAvxBackend::Quaternion::K(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst_k) {

    Util::CheckDuplicateArray(src, dst_k);

    if (length % 4 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckLength(length, src);
    Util::CheckLength(length / 4, dst_k);

    const float* src_ptr = (const float*)(src->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst_k->Ptr.ToPointer());

    quaternion_k(length, src_ptr, dst_ptr);
}