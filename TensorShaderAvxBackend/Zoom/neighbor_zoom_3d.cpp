#include "../TensorShaderAvxBackend.h"

using namespace System;

void neighbor_zoom_3d(unsigned int channels,
                      unsigned int inwidth, unsigned int inheight, unsigned int indepth,
                      unsigned int outwidth, unsigned int outheight, unsigned int outdepth,
                      unsigned int batch,
                      const float* __restrict inmap_ptr, float* __restrict outmap_ptr) {

    const unsigned int j = channels & ~7u, k = channels - j;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

    const unsigned int luf = 0, ruf = channels, ldf = channels * outwidth, rdf = channels * (1 + outwidth);
    const unsigned int lur = channels * outwidth * outheight, rur = ruf + lur, ldr = ldf + lur, rdr = rdf + lur;

    for (unsigned int th = 0; th < batch; th++) {

        for (unsigned int iz = 0, oz = 0; iz < indepth; iz++, oz += 2) {
            for (unsigned int iy = 0, oy = 0; iy < inheight; iy++, oy += 2) {
                for (unsigned int ix = 0, ox = 0; ix < inwidth; ix++, ox += 2) {

                    unsigned int inmap_idx = channels * (ix + inwidth * (iy + inheight * iz));
                    unsigned int outmap_idx = channels * (ox + outwidth * (oy + outheight * oz));

                    for (unsigned int i = 0; i < j; i += 8) {
                        __m256 x = _mm256_loadu_ps(inmap_ptr + inmap_idx + i);
                        _mm256_storeu_ps(outmap_ptr + outmap_idx + luf + i, x);
                        _mm256_storeu_ps(outmap_ptr + outmap_idx + ruf + i, x);
                        _mm256_storeu_ps(outmap_ptr + outmap_idx + ldf + i, x);
                        _mm256_storeu_ps(outmap_ptr + outmap_idx + rdf + i, x);
                        _mm256_storeu_ps(outmap_ptr + outmap_idx + lur + i, x);
                        _mm256_storeu_ps(outmap_ptr + outmap_idx + rur + i, x);
                        _mm256_storeu_ps(outmap_ptr + outmap_idx + ldr + i, x);
                        _mm256_storeu_ps(outmap_ptr + outmap_idx + rdr + i, x);
                    }
                    if (k > 0) {
                        __m256 x = _mm256_maskload_ps(inmap_ptr + inmap_idx + j, mask);
                        _mm256_maskstore_ps(outmap_ptr + outmap_idx + luf + j, mask, x);
                        _mm256_maskstore_ps(outmap_ptr + outmap_idx + ruf + j, mask, x);
                        _mm256_maskstore_ps(outmap_ptr + outmap_idx + ldf + j, mask, x);
                        _mm256_maskstore_ps(outmap_ptr + outmap_idx + rdf + j, mask, x);
                        _mm256_maskstore_ps(outmap_ptr + outmap_idx + lur + j, mask, x);
                        _mm256_maskstore_ps(outmap_ptr + outmap_idx + rur + j, mask, x);
                        _mm256_maskstore_ps(outmap_ptr + outmap_idx + ldr + j, mask, x);
                        _mm256_maskstore_ps(outmap_ptr + outmap_idx + rdr + j, mask, x);
                    }
                }
            }
        }

        inmap_ptr += channels * inwidth * inheight * indepth;
        outmap_ptr += channels * outwidth * outheight * outdepth;
    }
}

void TensorShaderAvxBackend::Zoom::NeighborZoom3D(unsigned int channels, unsigned int inwidth, unsigned int inheight, unsigned int indepth, unsigned int batch, AvxArray<float>^ inmap, AvxArray<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, outmap);

    unsigned int outwidth = inwidth * 2;
    unsigned int outheight = inheight * 2;
    unsigned int outdepth = indepth * 2;

    Util::CheckLength(channels * inwidth * inheight * indepth * batch, inmap);
    Util::CheckLength(channels * outwidth * outheight * outdepth * batch, outmap);

    const float* inmap_ptr = (const float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());

    neighbor_zoom_3d(channels, inwidth, inheight, indepth, outwidth, outheight, outdepth, batch, inmap_ptr, outmap_ptr);
}
