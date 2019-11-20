#include "../TensorShaderAvxBackend.h"

using namespace System;

void channel_to_space_3d(unsigned int inchannels, unsigned int outchannels,
    unsigned int inwidth, unsigned int inheight, unsigned int indepth,
    unsigned int outwidth, unsigned int outheight, unsigned int outdepth,
    unsigned int th, unsigned int scale, float* inmap_ptr, float* outmap_ptr) {

    const unsigned int inmap_offset = inchannels * inwidth * inheight * indepth * th, outmap_offset = outchannels * outwidth * outheight * outdepth * th;
    const unsigned int length = scale * outchannels;
    const unsigned int j = length & ~7u, k = length - j;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

    inmap_ptr += inmap_offset;
    outmap_ptr += outmap_offset;

    for (unsigned int iz = 0; iz < indepth; iz++) {
        for (unsigned int iy = 0; iy < inheight; iy++) {
            for (unsigned int ix = 0; ix < inwidth; ix++) {
                unsigned int inmap_idx = (ix + iy * inwidth + iz * inwidth * inheight) * inchannels;
                unsigned int outmap_org = (ix + iy * outwidth + iz * outwidth * outheight) * scale * outchannels;

                for (unsigned int kz = 0; kz < scale; kz++) {
                    unsigned int outmap_car = outmap_org;

                    for (unsigned int ky = 0; ky < scale; ky++) {
                        unsigned int outmap_idx = outmap_car;

                        for (unsigned int i = 0; i < j; i += 8) {
                            __m256 x = _mm256_loadu_ps(inmap_ptr + inmap_idx + i);
                            _mm256_storeu_ps(outmap_ptr + outmap_idx + i, x);
                        }
                        if (k > 0) {
                            __m256 x = _mm256_maskload_ps(inmap_ptr + inmap_idx + j, mask);
                            _mm256_maskstore_ps(outmap_ptr + outmap_idx + j, mask, x);
                        }

                        inmap_idx += length;
                        outmap_car += outwidth * outchannels;
                    }

                    outmap_org += outwidth * outheight * outchannels;
                }
            }
        }
    }
}

void TensorShaderAvxBackend::Transform::ChannelToSpace3D(unsigned int outchannels, unsigned int inwidth, unsigned int inheight, unsigned int indepth, unsigned int batch, unsigned int th, unsigned int scale, AvxArray<float>^ inmap, AvxArray<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, outmap);

    if (th >= batch) {
        throw gcnew System::ArgumentException();
    }

    unsigned int inchannels = outchannels * scale * scale * scale;
    unsigned int outwidth = inwidth * scale;
    unsigned int outheight = inheight * scale;
    unsigned int outdepth = indepth * scale;

    Util::CheckLength(inchannels * inwidth * inheight * indepth * batch, inmap);
    Util::CheckLength(outchannels * outwidth * outheight * outdepth * batch, outmap);

    float* inmap_ptr = (float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());

    channel_to_space_3d(inchannels, outchannels, inwidth, inheight, indepth, outwidth, outheight, outdepth, th, scale, inmap_ptr, outmap_ptr);
}
