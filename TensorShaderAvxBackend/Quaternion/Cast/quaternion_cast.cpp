#include "../../TensorShaderAvxBackend.h"

using namespace System;

void quaternion_cast(unsigned int length, 
                     const float* __restrict srcr_ptr, const float* __restrict srci_ptr, const float* __restrict srcj_ptr, const float* __restrict srck_ptr, float* __restrict dst_ptr) {

    for (unsigned int i = 0, j = 0; i < length; i += 4, j++) {
        dst_ptr[i] = srcr_ptr[j];
        dst_ptr[i + 1] = srci_ptr[j];
        dst_ptr[i + 2] = srcj_ptr[j];
        dst_ptr[i + 3] = srck_ptr[j];
    }
}

void TensorShaderAvxBackend::Quaternion::Cast(unsigned int length, AvxArray<float>^ src_r, AvxArray<float>^ src_i, AvxArray<float>^ src_j, AvxArray<float>^ src_k, AvxArray<float>^ dst) {

    Util::CheckDuplicateArray(src_r, src_i, src_j, src_k, dst);

    if (length % 4 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckLength(length / 4, src_r, src_i, src_j, src_k);
    Util::CheckLength(length, dst);
    
    const float* srcr_ptr = (const float*)(src_r->Ptr.ToPointer());
    const float* srci_ptr = (const float*)(src_i->Ptr.ToPointer());
    const float* srcj_ptr = (const float*)(src_j->Ptr.ToPointer());
    const float* srck_ptr = (const float*)(src_k->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    quaternion_cast(length, srcr_ptr, srci_ptr, srcj_ptr, srck_ptr, dst_ptr);
}