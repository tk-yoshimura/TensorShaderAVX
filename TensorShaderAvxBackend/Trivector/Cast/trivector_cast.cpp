#include "../../TensorShaderAvxBackend.h"

using namespace System;

void trivector_cast(unsigned int length, float* srcx_ptr, float* srcy_ptr, float* srcz_ptr, float* __restrict dst_ptr) {
    for (unsigned int i = 0, j = 0; i < length; i += 3, j++) {
        dst_ptr[i] = srcx_ptr[j];
        dst_ptr[i + 1] = srcy_ptr[j];
        dst_ptr[i + 2] = srcz_ptr[j];
    }
}

void TensorShaderAvxBackend::Trivector::Cast(unsigned int length, AvxArray<float>^ src_x, AvxArray<float>^ src_y, AvxArray<float>^ src_z, AvxArray<float>^ dst) {

    Util::CheckDuplicateArray(src_x, src_y, src_z, dst);

    if (length % 3 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckLength(length / 3, src_x, src_y, src_z);
    Util::CheckLength(length, dst);
    
    float* srcx_ptr = (float*)(src_x->Ptr.ToPointer());
    float* srcy_ptr = (float*)(src_y->Ptr.ToPointer());
    float* srcz_ptr = (float*)(src_z->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    trivector_cast(length, srcx_ptr, srcy_ptr, srcz_ptr, dst_ptr);
}