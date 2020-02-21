#include "../TensorShaderAvxBackend.h"

using namespace System;

void neighbor_zoom_1d(unsigned int channels,
                      unsigned int inwidth, unsigned int outwidth,
                      unsigned int batch,
                      const float* __restrict inmap_ptr, float* __restrict outmap_ptr) {

    const unsigned int j = channels & ~7u, k = channels - j;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

    const unsigned int l = 0, r = channels;

    for (unsigned int th = 0; th < batch; th++) {
 
        for (unsigned int ix = 0, ox = 0; ix < inwidth; ix++, ox += 2) {

            unsigned int inmap_idx = channels * ix;
            unsigned int outmap_idx = channels * ox;

            for (unsigned int i = 0; i < j; i += 8) {
                __m256 x = _mm256_loadu_ps(inmap_ptr + inmap_idx + i);
                _mm256_storeu_ps(outmap_ptr + outmap_idx + l + i, x);
                _mm256_storeu_ps(outmap_ptr + outmap_idx + r + i, x);
            }
            if (k > 0) {
                __m256 x = _mm256_maskload_ps(inmap_ptr + inmap_idx + j, mask);
                _mm256_maskstore_ps(outmap_ptr + outmap_idx + l + j, mask, x);
                _mm256_maskstore_ps(outmap_ptr + outmap_idx + r + j, mask, x);
            }
        }

        inmap_ptr += channels * inwidth;
        outmap_ptr += channels * outwidth;
    }
}

void TensorShaderAvxBackend::Zoom::NeighborZoom1D(unsigned int channels, unsigned int inwidth, unsigned int batch, AvxArray<float>^ inmap, AvxArray<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, outmap);

    unsigned int outwidth = inwidth * 2;

    Util::CheckLength(channels * inwidth * batch, inmap);
    Util::CheckLength(channels * outwidth * batch, outmap);

    const float* inmap_ptr = (const float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());

    neighbor_zoom_1d(channels, inwidth, outwidth, batch, inmap_ptr, outmap_ptr);
}
