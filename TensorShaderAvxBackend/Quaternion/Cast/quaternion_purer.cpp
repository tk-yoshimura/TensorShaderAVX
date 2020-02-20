#include "../../TensorShaderAvxBackend.h"

using namespace System;

void quaternion_purer(unsigned int length, float* srcr_ptr, float* __restrict dst_ptr) {
    for (unsigned int i = 0, j = 0; i < length; i += 4, j++) {
        dst_ptr[i] = srcr_ptr[j];
        dst_ptr[i + 1] = dst_ptr[i + 2] = dst_ptr[i + 3] = 0;
    }
}

void TensorShaderAvxBackend::Quaternion::PureR(unsigned int length, AvxArray<float>^ src_r, AvxArray<float>^ dst) {

    Util::CheckDuplicateArray(src_r, dst);

    if (length % 4 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckLength(length / 4, src_r);
    Util::CheckLength(length, dst);

    float* srcr_ptr = (float*)(src_r->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    quaternion_purer(length, srcr_ptr, dst_ptr);
}