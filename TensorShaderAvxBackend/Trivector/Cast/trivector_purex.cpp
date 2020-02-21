#include "../../TensorShaderAvxBackend.h"

using namespace System;

void trivector_purex(unsigned int length, const float* __restrict srcx_ptr, float* __restrict dst_ptr) {

    for (unsigned int i = 0, j = 0; i < length; i += 3, j++) {
        dst_ptr[i] = srcx_ptr[j];
        dst_ptr[i + 1] = 0;
        dst_ptr[i + 2] = 0;
    }
}

void TensorShaderAvxBackend::Trivector::PureX(unsigned int length, AvxArray<float>^ src_x, AvxArray<float>^ dst) {

    Util::CheckDuplicateArray(src_x, dst);

    if (length % 3 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckLength(length / 3, src_x);
    Util::CheckLength(length, dst);
    
    const float* srcx_ptr = (const float*)(src_x->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    trivector_purex(length, srcx_ptr, dst_ptr);
}