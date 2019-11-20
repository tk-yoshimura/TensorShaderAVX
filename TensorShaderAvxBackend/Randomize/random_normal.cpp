#include "../TensorShaderAvxBackend.h"

#include <math.h>

using namespace System;

void random_normal(unsigned int length, unsigned __int64 seed, float* dst_ptr) {
    const double inv = 1.0 / 0x100000000ull, pi2 = 6.2831853071795865;

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

        double u = x.u32[0] * inv, v = x.u32[1] * inv;
        double sq_log_u = sqrt(-2.0 * log(1 - u)), pi2_v = pi2 * v;

        dst_ptr[i] = (float)(sq_log_u * sin(pi2_v));
        dst_ptr[i+1] = (float)(sq_log_u * cos(pi2_v));
    }

    if ((length & 1) > 0) {
        x.u64 = x.u64 ^ (x.u64 << 7);
        x.u64 = x.u64 ^ (x.u64 >> 9);

        double u = x.u32[0] * inv, v = x.u32[1] * inv;
        double sq_log_u = sqrt(-2.0 * log(u)), pi2_v = pi2 * v;

        dst_ptr[length & ~1] = (float)(sq_log_u * sin(pi2_v));
    }
}

void TensorShaderAvxBackend::Randomize::Normal(unsigned int length, AvxArray<float>^ dst, Random^ random) {

    Util::CheckLength(length, dst);

    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    unsigned __int64 seed = ((unsigned __int64)random->Next() + 1) * 0x100000000ull + ((unsigned __int64)random->Next() + 1);

    random_normal(length, seed, dst_ptr);
}
