#include "../../TensorShaderAvxBackend.h"

using namespace System;

void quaternion_purei(unsigned int length, float* srci_ptr, float* dst_ptr) {
    for (unsigned int i = 0, j = 0; i < length; i += 4, j++) {
        dst_ptr[i + 1] = srci_ptr[j];
        dst_ptr[i] = dst_ptr[i + 2] = dst_ptr[i + 3] = 0;
    }
}

void TensorShaderAvxBackend::Quaternion::PureI(unsigned int length, AvxArray<float>^ src_i, AvxArray<float>^ dst) {

    Util::CheckDuplicateArray(src_i, dst);

    if (length % 4 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckLength(length / 4, src_i);
    Util::CheckLength(length, dst);

    float* srci_ptr = (float*)(src_i->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    quaternion_purei(length, srci_ptr, dst_ptr);
}