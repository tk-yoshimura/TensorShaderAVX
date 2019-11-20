#include "../TensorShaderAvxBackend.h"

using namespace System;

void trimming_2d(unsigned int channels,
                 unsigned int inwidth, unsigned int inheight,
                 unsigned int outwidth, unsigned int outheight,
                 unsigned int th,
                 unsigned int trim_left, unsigned int trim_right,
                 unsigned int trim_top, unsigned int trim_bottom,
                 float* inmap_ptr, float* outmap_ptr) {

    const unsigned int inmap_offset = channels * inwidth * inheight * th, outmap_offset = channels * outwidth * outheight * th;
    inmap_ptr += inmap_offset;
    outmap_ptr += outmap_offset;

    const unsigned int length = channels * outwidth;
    const unsigned int j = length & ~7u, k = length - j;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

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
}

void TensorShaderAvxBackend::Trimming::Trimming2D(unsigned int channels, unsigned int outwidth, unsigned int outheight, unsigned int batch, unsigned int th, unsigned int trim_left, unsigned int trim_right, unsigned int trim_top, unsigned int trim_bottom, AvxArray<float>^ inmap, AvxArray<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, outmap);

    if (th >= batch) {
        throw gcnew System::ArgumentException();
    }

    unsigned int inwidth = outwidth + trim_left + trim_right;
    unsigned int inheight = outheight + trim_top + trim_bottom;

    Util::CheckLength(channels * inwidth * inheight * batch, inmap);
    Util::CheckLength(channels * outwidth * outheight * batch, outmap);

    float* inmap_ptr = (float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());

    trimming_2d(channels, inwidth, inheight, outwidth, outheight, th, trim_left, trim_right, trim_top, trim_bottom, inmap_ptr, outmap_ptr);
}
