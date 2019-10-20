#include "../TensorShaderAvxBackend.h"

using namespace System;

void trimming_3d(unsigned int channels,
                 unsigned int inwidth, unsigned int inheight, unsigned int indepth,
                 unsigned int outwidth, unsigned int outheight, unsigned int outdepth,
                 unsigned int th,
                 unsigned int trim_left, unsigned int trim_right,
                 unsigned int trim_top, unsigned int trim_bottom,
                 unsigned int trim_front, unsigned int trim_rear,
                 float* inmap_ptr, float* outmap_ptr) {

    const unsigned int inmap_offset = channels * inwidth * inheight * indepth * th;
    const unsigned int outmap_offset = channels * outwidth * outheight * outdepth * th;
    inmap_ptr += inmap_offset;
    outmap_ptr += outmap_offset;
    
    for (unsigned int oz = 0; oz < outdepth; oz++) {
        const unsigned int length = channels * outwidth;
        const unsigned int j = length & ~7u, k = length - j;
        const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        unsigned int inmap_idx = channels * (trim_left + inwidth * (trim_top + inheight * (oz + trim_front)));
        unsigned int outmap_idx = channels * outwidth * outheight * oz;

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
}

void TensorShaderAvxBackend::Trimming::Trimming3D(unsigned int channels, unsigned int outwidth, unsigned int outheight, unsigned int outdepth, unsigned int batch, unsigned int th, unsigned int trim_left, unsigned int trim_right, unsigned int trim_top, unsigned int trim_bottom, unsigned int trim_front, unsigned int trim_rear, cli::array<float>^ inmap, cli::array<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, outmap);

    unsigned int inwidth = outwidth + trim_left + trim_right;
    unsigned int inheight = outheight + trim_top + trim_bottom;
    unsigned int indepth = outdepth + trim_front + trim_rear;

    if (channels * inwidth * inheight * indepth * batch > (unsigned int)inmap->Length) {
        throw gcnew System::ArgumentException();
    }
    if (channels * outwidth * outheight * outdepth * batch > (unsigned int)outmap->Length) {
        throw gcnew System::ArgumentException();
    }
    if (th >= batch) {
        throw gcnew System::ArgumentException();
    }

    pin_ptr<float> pinptr_inmap = &inmap[0];
    pin_ptr<float> pinptr_outmap = &outmap[0];

    float* inmap_ptr = pinptr_inmap;
    float* outmap_ptr = pinptr_outmap;

    trimming_3d(channels, inwidth, inheight, indepth, outwidth, outheight, outdepth, th, trim_left, trim_right, trim_top, trim_bottom, trim_front, trim_rear, inmap_ptr, outmap_ptr);
}
