#include "../../TensorShaderAvxBackend.h"

using namespace System;

void quaternion_cast(unsigned int length, float* srcr_ptr, float* srci_ptr, float* srcj_ptr, float* srck_ptr, float* dst_ptr) {
    for (unsigned int i = 0, j = 0; i < length; i += 4, j++) {
        dst_ptr[i] = srcr_ptr[j];
        dst_ptr[i + 1] = srci_ptr[j];
        dst_ptr[i + 2] = srcj_ptr[j];
        dst_ptr[i + 3] = srck_ptr[j];
    }
}

void TensorShaderAvxBackend::Quaternion::Cast(unsigned int length, cli::array<float>^ src_r, cli::array<float>^ src_i, cli::array<float>^ src_j, cli::array<float>^ src_k, cli::array<float>^ dst) {

    Util::CheckDuplicateArray(src_r, src_i, src_j, src_k, dst);

    if (length % 4 != 0) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckOutOfRange(0, length / 4, src_r, src_i, src_j, src_k);
    Util::CheckOutOfRange(0, length, dst);

    pin_ptr<float> pinptr_srcr = &src_r[0];
    pin_ptr<float> pinptr_srci = &src_i[0];
    pin_ptr<float> pinptr_srcj = &src_j[0];
    pin_ptr<float> pinptr_srck = &src_k[0];
    pin_ptr<float> pinptr_dst = &dst[0];

    float* srcr_ptr = pinptr_srcr;
    float* srci_ptr = pinptr_srci;
    float* srcj_ptr = pinptr_srcj;
    float* srck_ptr = pinptr_srck;
    float* dst_ptr = pinptr_dst;

    quaternion_cast(length, srcr_ptr, srci_ptr, srcj_ptr, srck_ptr, dst_ptr);
}