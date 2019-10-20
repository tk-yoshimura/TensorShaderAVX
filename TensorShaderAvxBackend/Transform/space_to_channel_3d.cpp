#include "../TensorShaderAvxBackend.h"

using namespace System;

void space_to_channel_3d(unsigned int inchannels, unsigned int outchannels,
    unsigned int inwidth, unsigned int inheight, unsigned int indepth,
    unsigned int outwidth, unsigned int outheight, unsigned int outdepth,
    unsigned int th, unsigned int scale, float* inmap_ptr, float* outmap_ptr) {

    const unsigned int inmap_offset = inchannels * inwidth * inheight * indepth * th, outmap_offset = outchannels * outwidth * outheight * outdepth * th;
    const unsigned int length = scale * inchannels;
    const unsigned int j = length & ~7u, k = length - j;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

    inmap_ptr += inmap_offset;
    outmap_ptr += outmap_offset;

    for (unsigned int oz = 0; oz < outdepth; oz++) {
        for (unsigned int oy = 0; oy < outheight; oy++) {
            for (unsigned int ox = 0; ox < outwidth; ox++) {
                unsigned int inmap_org = (ox + oy * inwidth + oz * inwidth * inheight) * scale * inchannels;
                unsigned int outmap_idx = (ox + oy * outwidth + oz * outwidth * outheight) * outchannels;

                for (unsigned int kz = 0; kz < scale; kz++) {
                    unsigned int inmap_car = inmap_org;

                    for (unsigned int ky = 0; ky < scale; ky++) {
                        unsigned int inmap_idx = inmap_car;

                        for (unsigned int i = 0; i < j; i += 8) {
                            __m256 x = _mm256_loadu_ps(inmap_ptr + inmap_idx + i);
                            _mm256_storeu_ps(outmap_ptr + outmap_idx + i, x);
                        }
                        if (k > 0) {
                            __m256 x = _mm256_maskload_ps(inmap_ptr + inmap_idx + j, mask);
                            _mm256_maskstore_ps(outmap_ptr + outmap_idx + j, mask, x);
                        }

                        outmap_idx += length;
                        inmap_car += inwidth * inchannels;
                    }
                    inmap_org += inwidth * inheight * inchannels;
                }
            }
        }
    }
}

void TensorShaderAvxBackend::Transform::SpaceToChannel3D(unsigned int inchannels, unsigned int outwidth, unsigned int outheight, unsigned int outdepth, unsigned int batch, unsigned int th, unsigned int scale, cli::array<float>^ inmap, cli::array<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, outmap);

    unsigned int outchannels = inchannels * scale * scale * scale;
    unsigned int inwidth = outwidth * scale, inheight = outheight * scale, indepth = outdepth * scale;

    if (inchannels * inwidth * inheight * indepth * batch > (unsigned int)inmap->Length) {
        throw gcnew System::ArgumentException();
    }
    if (outchannels * outwidth * outheight * outdepth * batch > (unsigned int)outmap->Length) {
        throw gcnew System::ArgumentException();
    }
    if (th >= batch) {
        throw gcnew System::ArgumentException();
    }

    pin_ptr<float> pinptr_inmap = &inmap[0];
    pin_ptr<float> pinptr_outmap = &outmap[0];

    float* inmap_ptr = pinptr_inmap;
    float* outmap_ptr = pinptr_outmap;

    space_to_channel_3d(inchannels, outchannels, inwidth, inheight, indepth, outwidth, outheight, outdepth, th, scale, inmap_ptr, outmap_ptr);
}
