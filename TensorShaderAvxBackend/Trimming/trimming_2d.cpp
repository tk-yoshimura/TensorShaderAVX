#include "../TensorShaderAvxBackend.h"

using namespace System;

void trimming_2d(unsigned int channels,
                 unsigned int inwidth, unsigned int inheight,
                 unsigned int outwidth, unsigned int outheight,
                 unsigned int batch,
                 unsigned int trim_left, unsigned int trim_right,
                 unsigned int trim_top, unsigned int trim_bottom,
                 const float* __restrict inmap_ptr, float* __restrict outmap_ptr) {

    const unsigned int length = channels * outwidth;
    const unsigned int j = length & ~7u, k = length - j;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

    for (unsigned int th = 0; th < batch; th++) {

        unsigned int inmap_idx = channels * (trim_left + inwidth * trim_top);
        unsigned int outmap_idx = 0;

        for (unsigned int oy = 0; oy < outheight; oy++) {
            for (unsigned int i = 0; i < j; i += 8) {
                __m256 x = _mm256_loadu_ps(inmap_ptr + inmap_idx + i);
                _mm256_storeu_ps(outmap_ptr + outmap_idx + i, x);
            }
            if (k > 0) {
                __m256 x = _mm256_maskload_ps(inmap_ptr + inmap_idx + j, mask);
                _mm256_maskstore_ps(outmap_ptr + outmap_idx + j, mask, x);
            }

            inmap_idx += channels * inwidth;
            outmap_idx += channels * outwidth;
        }

        inmap_ptr += channels * inwidth * inheight;
        outmap_ptr += channels * outwidth * outheight;
    }
}

void TensorShaderAvxBackend::Trimming::Trimming2D(unsigned int channels, unsigned int outwidth, unsigned int outheight, unsigned int batch, unsigned int trim_left, unsigned int trim_right, unsigned int trim_top, unsigned int trim_bottom, AvxArray<float>^ inmap, AvxArray<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, outmap);

    unsigned int inwidth = outwidth + trim_left + trim_right;
    unsigned int inheight = outheight + trim_top + trim_bottom;

    Util::CheckLength(channels * inwidth * inheight * batch, inmap);
    Util::CheckLength(channels * outwidth * outheight * batch, outmap);

    const float* inmap_ptr = (const float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());

    trimming_2d(channels, 
                inwidth, inheight, 
                outwidth, outheight, 
                batch, 
                trim_left, trim_right, trim_top, trim_bottom, 
                inmap_ptr, outmap_ptr);
}
