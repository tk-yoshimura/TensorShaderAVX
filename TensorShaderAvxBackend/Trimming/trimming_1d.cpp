#include "../TensorShaderAvxBackend.h"

using namespace System;

void trimming_1d(unsigned int channels, 
                 unsigned int inwidth, 
                 unsigned int outwidth, 
                 unsigned int batch, 
                 unsigned int trim_left, unsigned int trim_right, 
                 const float* __restrict inmap_ptr, float* __restrict outmap_ptr) {
    
    const unsigned int length = channels * outwidth;
    const unsigned int j = length & ~7u, k = length - j;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

    for (unsigned int th = 0; th < batch; th++) {

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

        inmap_ptr += channels * inwidth;
        outmap_ptr += channels * outwidth;
    }
}

void TensorShaderAvxBackend::Trimming::Trimming1D(unsigned int channels, unsigned int outwidth, unsigned int batch, unsigned int trim_left, unsigned int trim_right, AvxArray<float>^ inmap, AvxArray<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, outmap);

    unsigned int inwidth = outwidth + trim_left + trim_right;

    Util::CheckLength(channels * inwidth * batch, inmap);
    Util::CheckLength(channels * outwidth * batch, outmap);

    const float* inmap_ptr = (const float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());

    trimming_1d(channels, 
                inwidth, outwidth, 
                batch, 
                trim_left, trim_right, 
                inmap_ptr, outmap_ptr);
}
