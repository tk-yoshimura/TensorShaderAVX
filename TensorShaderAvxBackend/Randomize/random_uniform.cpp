#include "../TensorShaderAvxBackend.h"

using namespace System;

void random_uniform(unsigned int length, unsigned __int64 seed, float* dst_ptr) {
    const double inv = 1.0 / 0xFFFFFFFFu;

    union {
        unsigned __int64 u64;
        unsigned __int32 u32[2];
    } x;

    x.u64 = seed;

    for (unsigned int i = 0; i < 8; i++) {
        x.u64 = x.u64 ^ (x.u64 << 7);
        x.u64 = x.u64 ^ (x.u64 >> 9);
    }

    for (unsigned int i = 0; i < (length & ~1); i += 2) {
        x.u64 = x.u64 ^ (x.u64 << 7);
        x.u64 = x.u64 ^ (x.u64 >> 9);

        dst_ptr[i] = (float)(x.u32[0] * inv);
        dst_ptr[i + 1] = (float)(x.u32[1] * inv);
    }

    if ((length & 1) > 0) {
        x.u64 = x.u64 ^ (x.u64 << 7);
        x.u64 = x.u64 ^ (x.u64 >> 9);

        dst_ptr[length & ~1] = (float)(x.u32[0] * inv);
    }
}

void TensorShaderAvxBackend::Randomize::Uniform(unsigned int length, AvxArray<float>^ dst, Random^ random) {

    Util::CheckLength(length, dst);

    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    unsigned __int64 seed = ((unsigned __int64)random->Next() + 1) * 0x100000000ull + ((unsigned __int64)random->Next() + 1);

    random_uniform(length, seed, dst_ptr);
}
