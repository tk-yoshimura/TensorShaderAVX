#include "../../TensorShaderAvxBackend.h"

using namespace System;

void quaternion_r(unsigned int length, const float* __restrict src_ptr, float* __restrict dst_ptr) {
    for (unsigned int i = 0, j = 0; i < length; i += 4, j++) {
        dst_ptr[j] = src_ptr[i];
    }
}

void TensorShaderAvxBackend::Quaternion::R(unsigned int length, AvxArray<float>^ src, AvxArray<float>^ dst_r) {

    Util::CheckDuplicateArray(src, dst_r);

    if (length % 4 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckLength(length, src);
    Util::CheckLength(length / 4, dst_r);

    const float* src_ptr = (const float*)(src->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst_r->Ptr.ToPointer());

    quaternion_r(length, src_ptr, dst_ptr);
}