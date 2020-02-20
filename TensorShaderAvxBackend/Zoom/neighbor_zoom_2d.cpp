#include "../TensorShaderAvxBackend.h"

using namespace System;

void neighbor_zoom_2d(unsigned int channels,
                      unsigned int inwidth, unsigned int inheight, 
                      unsigned int outwidth, unsigned int outheight,
                      unsigned int th,
                      const float* __restrict inmap_ptr, float* outmap_ptr) {

    const unsigned int inmap_offset = channels * inwidth * inheight * th, outmap_offset = channels * outwidth * outheight * th;
    inmap_ptr += inmap_offset;
    outmap_ptr += outmap_offset;

    const unsigned int j = channels & ~7u, k = channels - j;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

    const unsigned int lu = 0, ru = channels, ld = channels * outwidth, rd = channels * (1 + outwidth);

    for (unsigned int iy = 0, oy = 0; iy < inheight; iy++, oy += 2) {
        for (unsigned int ix = 0, ox = 0; ix < inwidth; ix++, ox += 2) {

            unsigned int inmap_idx = channels * (ix + inwidth * iy);
            unsigned int outmap_idx = channels * (ox + outwidth * oy);
             
            for (unsigned int i = 0; i < j; i += 8) {
                __m256 x = _mm256_loadu_ps(inmap_ptr + inmap_idx + i);
                _mm256_storeu_ps(outmap_ptr + outmap_idx + lu + i, x);
                _mm256_storeu_ps(outmap_ptr + outmap_idx + ru + i, x);
                _mm256_storeu_ps(outmap_ptr + outmap_idx + ld + i, x);
                _mm256_storeu_ps(outmap_ptr + outmap_idx + rd + i, x);
            }
            if (k > 0) {
                __m256 x = _mm256_maskload_ps(inmap_ptr + inmap_idx + j, mask);
                _mm256_maskstore_ps(outmap_ptr + outmap_idx + lu + j, mask, x);
                _mm256_maskstore_ps(outmap_ptr + outmap_idx + ru + j, mask, x);
                _mm256_maskstore_ps(outmap_ptr + outmap_idx + ld + j, mask, x);
                _mm256_maskstore_ps(outmap_ptr + outmap_idx + rd + j, mask, x);
            }
        }
    }
}

void TensorShaderAvxBackend::Zoom::NeighborZoom2D(unsigned int channels, unsigned int inwidth, unsigned int inheight, unsigned int batch, unsigned int th, AvxArray<float>^ inmap, AvxArray<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, outmap);

    if (th >= batch) {
        throw gcnew System::ArgumentException();
    }

    unsigned int outwidth = inwidth * 2;
    unsigned int outheight = inheight * 2;

    Util::CheckLength(channels * inwidth * inheight * batch, inmap);
    Util::CheckLength(channels * outwidth * outheight * batch, outmap);

    float* inmap_ptr = (float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());

    neighbor_zoom_2d(channels, inwidth, inheight, outwidth, outheight, th, inmap_ptr, outmap_ptr);
}
