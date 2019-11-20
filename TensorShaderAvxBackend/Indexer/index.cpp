#include "../TensorShaderAvxBackend.h"

using namespace System;

void index(unsigned int stride, unsigned int axislength, unsigned int clones, float* dst_ptr) {
    for (unsigned int k = 0, idx = 0; k < clones; k++) {
        for (unsigned int j = 0; j < axislength; j++) {
            for (unsigned int i = 0; i < stride; i++, idx++) {
                dst_ptr[idx] = (float)j;
            }
        }
    }
}

void TensorShaderAvxBackend::Indexer::Index(unsigned int stride, unsigned int axislength, unsigned int clones, AvxArray<float>^ dst) {

    Util::CheckLength(stride * axislength * clones, dst);

    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    index(stride, axislength, clones, dst_ptr);
}