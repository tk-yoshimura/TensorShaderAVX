#include "../TensorShaderAvxBackend.h"

using namespace System;

void space_to_channel_2d(unsigned int inchannels, unsigned int outchannels,
    unsigned int inwidth, unsigned int inheight,
    unsigned int outwidth, unsigned int outheight,
    unsigned int th, unsigned int scale, const float* __restrict inmap_ptr, float* __restrict outmap_ptr) {

    const unsigned int inmap_offset = inchannels * inwidth * inheight * th, outmap_offset = outchannels * outwidth * outheight * th;
    const unsigned int length = scale * inchannels;
    const unsigned int j = length & ~7u, k = length - j;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

    inmap_ptr += inmap_offset;
    outmap_ptr += outmap_offset;

    for (unsigned int oy = 0; oy < outheight; oy++) {
        for (unsigned int ox = 0; ox < outwidth; ox++) {
            unsigned int inmap_org = (ox + oy * inwidth) * scale * inchannels;
            unsigned int outmap_idx = (ox + oy * outwidth) * outchannels;

            for (unsigned int ky = 0; ky < scale; ky++) {
                unsigned int inmap_idx = inmap_org;

                for (unsigned int i = 0; i < j; i += 8) {
                    __m256 x = _mm256_loadu_ps(inmap_ptr + inmap_idx + i);
                    _mm256_storeu_ps(outmap_ptr + outmap_idx + i, x);
                }
                if (k > 0) {
                    __m256 x = _mm256_maskload_ps(inmap_ptr + inmap_idx + j, mask);
                    _mm256_maskstore_ps(outmap_ptr + outmap_idx + j, mask, x);
                }

                outmap_idx += length;
                inmap_org += inwidth * inchannels;
            }
        }
    }
}

void TensorShaderAvxBackend::Transform::SpaceToChannel2D(unsigned int inchannels, unsigned int outwidth, unsigned int outheight, unsigned int batch, unsigned int th, unsigned int scale, AvxArray<float>^ inmap, AvxArray<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, outmap);

    if (th >= batch) {
        throw gcnew System::ArgumentException();
    }

    unsigned int outchannels = inchannels * scale * scale;
    unsigned int inwidth = outwidth * scale;
    unsigned int inheight = outheight * scale;

    Util::CheckLength(inchannels * inwidth * inheight * batch, inmap);
    Util::CheckLength(outchannels * outwidth * outheight * batch, outmap);

    const float* inmap_ptr = (const float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());

    space_to_channel_2d(inchannels, outchannels, inwidth, inheight, outwidth, outheight, th, scale, inmap_ptr, outmap_ptr);
}
