#include "../TensorShaderAvxBackend.h"

using namespace System;

void trimming_1d(unsigned int channels, 
                 unsigned int inwidth, 
                 unsigned int outwidth, 
                 unsigned int th, 
                 unsigned int trim_left, unsigned int trim_right, 
                 const float* __restrict inmap_ptr, float* outmap_ptr) {

    const unsigned int inmap_offset = channels * inwidth * th, outmap_offset = channels * outwidth * th;
    inmap_ptr += inmap_offset;
    outmap_ptr += outmap_offset;

    const unsigned int length = channels * outwidth;
    const unsigned int j = length & ~7u, k = length - j;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

    const unsigned int inmap_idx = channels * trim_left;
    const unsigned int outmap_idx = 0;

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x = _mm256_loadu_ps(inmap_ptr + inmap_idx + i);
        _mm256_storeu_ps(outmap_ptr + outmap_idx + i, x);
    }
    if (k > 0) {
        __m256 x = _mm256_maskload_ps(inmap_ptr + inmap_idx + j, mask);
        _mm256_maskstore_ps(outmap_ptr + outmap_idx + j, mask, x);
    }
}

void TensorShaderAvxBackend::Trimming::Trimming1D(unsigned int channels, unsigned int outwidth, unsigned int batch, unsigned int th, unsigned int trim_left, unsigned int trim_right, AvxArray<float>^ inmap, AvxArray<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, outmap);

    if (th >= batch) {
        throw gcnew System::ArgumentException();
    }

    unsigned int inwidth = outwidth + trim_left + trim_right;

    Util::CheckLength(channels * inwidth * batch, inmap);
    Util::CheckLength(channels * outwidth * batch, outmap);

    float* inmap_ptr = (float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());

    trimming_1d(channels, inwidth, outwidth, th, trim_left, trim_right, inmap_ptr, outmap_ptr);
}
