#include "../../TensorShaderAvxBackend.h"

using namespace System;

void quaternion_purek(unsigned int length, const float* __restrict srck_ptr, float* __restrict dst_ptr) {
    for (unsigned int i = 0, j = 0; i < length; i += 4, j++) {
        dst_ptr[i + 3] = srck_ptr[j];
        dst_ptr[i + 0] = dst_ptr[i + 1] = dst_ptr[i + 2] = 0;
    }
}

void TensorShaderAvxBackend::Quaternion::PureK(unsigned int length, AvxArray<float>^ src_k, AvxArray<float>^ dst) {

    Util::CheckDuplicateArray(src_k, dst);

    if (length % 4 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckLength(length / 4, src_k);
    Util::CheckLength(length, dst);

    const float* srck_ptr = (const float*)(src_k->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    quaternion_purek(length, srck_ptr, dst_ptr);
}