#include "../../TensorShaderAvxBackend.h"

using namespace System;

void trivector_purez(unsigned int length, const float* __restrict srcz_ptr, float* __restrict dst_ptr) {

    for (unsigned int i = 0, j = 0; i < length; i += 3, j++) {
        dst_ptr[i] = 0;
        dst_ptr[i + 1] = 0;
        dst_ptr[i + 2] = srcz_ptr[j];
    }
}

void TensorShaderAvxBackend::Trivector::PureZ(unsigned int length, AvxArray<float>^ src_z, AvxArray<float>^ dst) {

    Util::CheckDuplicateArray(src_z, dst);

    if (length % 3 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckLength(length / 3, src_z);
    Util::CheckLength(length, dst);
    
    const float* srcz_ptr = (const float*)(src_z->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    trivector_purez(length, srcz_ptr, dst_ptr);
}