#include "../TensorShaderAvxBackend.h"

using namespace System;

void random_bernoulli(unsigned int length, unsigned __int64 seed, double prob, float* dst_ptr) {
    const unsigned __int64 thr = (unsigned __int64)((double)(0x1000000000000ull) * prob);
     
    unsigned __int64 x = seed;

    for (unsigned int i = 0; i < 8; i++) {
        x = x ^ (x << 7);
        x = x ^ (x >> 9);
    }

    for (unsigned int i = 0; i < length; i++) {
        x = x ^ (x << 7);
        x = x ^ (x >> 9);

        dst_ptr[i] = (x >> 16) < thr ? (float)1 : (float)0;
    }
}

void TensorShaderAvxBackend::Randomize::Bernoulli(unsigned int length, double prob, AvxArray<float>^ dst, Random^ random) {

    Util::CheckLength(length, dst);

    if (prob < 0 || !(prob <= 1)) {
        throw gcnew System::ArgumentException();
    }

    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    unsigned __int64 seed = ((unsigned __int64)random->Next() + 1) * 0x100000000ull + ((unsigned __int64)random->Next() + 1);

    random_bernoulli(length, seed, prob, dst_ptr);
}
