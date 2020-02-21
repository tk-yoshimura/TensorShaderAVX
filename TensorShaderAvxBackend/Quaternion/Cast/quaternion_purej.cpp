#include "../../TensorShaderAvxBackend.h"

using namespace System;

void quaternion_purej(unsigned int length, const float* __restrict srcj_ptr, float* __restrict dst_ptr) {
    for (unsigned int i = 0, j = 0; i < length; i += 4, j++) {
        dst_ptr[i + 2] = srcj_ptr[j];
        dst_ptr[i] = dst_ptr[i + 1] = dst_ptr[i + 3] = 0;
    }
}

void TensorShaderAvxBackend::Quaternion::PureJ(unsigned int length, AvxArray<float>^ src_j, AvxArray<float>^ dst) {

    Util::CheckDuplicateArray(src_j, dst);

    if (length % 4 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckLength(length / 4, src_j);
    Util::CheckLength(length, dst);

    const float* srcj_ptr = (const float*)(src_j->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    quaternion_purej(length, srcj_ptr, dst_ptr);
}