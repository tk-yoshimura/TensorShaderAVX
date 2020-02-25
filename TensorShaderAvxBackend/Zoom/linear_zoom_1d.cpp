#include "../TensorShaderAvxBackend.h"

using namespace System;

void linear_zoom_1d(unsigned int channels,
                    unsigned int inwidth, unsigned int outwidth,
                    unsigned int batch,
                    const float* __restrict inmap_ptr, float* __restrict outmap_ptr) {
    
    const unsigned int j = channels & ~7u, k = channels - j;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);
    
    const __m256 cinv3 = _mm256_set1_ps(float(1.0 / 3.0));
    const __m256 c2 = _mm256_set1_ps(2);

    const unsigned int l = 0, r = channels;

    for (unsigned int th = 0; th < batch; th++) {

        for (unsigned int ixc = 0, ox = 0; ixc < inwidth; ixc++, ox += 2) {
            unsigned int ixl = ((ixc > 0) ? ixc - 1 : 0);
            unsigned int ixr = ((ixc < inwidth - 1) ? ixc + 1 : inwidth - 1);

            unsigned int inmap_c_idx = channels * ixc;
            unsigned int inmap_l_idx = channels * ixl;
            unsigned int inmap_r_idx = channels * ixr;

            unsigned int outmap_idx = channels * ox;

            for (unsigned int i = 0; i < j; i += 8) {
                __m256 xc = _mm256_mul_ps(_mm256_loadu_ps(inmap_ptr + inmap_c_idx + i), c2);
                __m256 xl = _mm256_loadu_ps(inmap_ptr + inmap_l_idx + i);
                __m256 xr = _mm256_loadu_ps(inmap_ptr + inmap_r_idx + i);

                __m256 yl = _mm256_mul_ps(cinv3, _mm256_add_ps(xc, xl));
                __m256 yr = _mm256_mul_ps(cinv3, _mm256_add_ps(xc, xr));

                _mm256_storeu_ps(outmap_ptr + outmap_idx + l + i, yl);
                _mm256_storeu_ps(outmap_ptr + outmap_idx + r + i, yr);
            }
            if (k > 0) {
                __m256 xc = _mm256_mul_ps(_mm256_maskload_ps(inmap_ptr + inmap_c_idx + j, mask), c2);
                __m256 xl = _mm256_maskload_ps(inmap_ptr + inmap_l_idx + j, mask);
                __m256 xr = _mm256_maskload_ps(inmap_ptr + inmap_r_idx + j, mask);

                __m256 yl = _mm256_mul_ps(cinv3, _mm256_add_ps(xc, xl));
                __m256 yr = _mm256_mul_ps(cinv3, _mm256_add_ps(xc, xr));

                _mm256_maskstore_ps(outmap_ptr + outmap_idx + l + j, mask, yl);
                _mm256_maskstore_ps(outmap_ptr + outmap_idx + r + j, mask, yr);
            }
        }

        inmap_ptr += channels * inwidth;
        outmap_ptr += channels * outwidth;
    }
}

void TensorShaderAvxBackend::Zoom::LinearZoom1D(unsigned int channels, unsigned int inwidth, unsigned int batch, AvxArray<float>^ inmap, AvxArray<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, outmap);

    unsigned int outwidth = inwidth * 2;

    Util::CheckLength(channels * inwidth * batch, inmap);
    Util::CheckLength(channels * outwidth * batch, outmap);

    const float* inmap_ptr = (const float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());

    linear_zoom_1d(channels, inwidth, outwidth, batch, inmap_ptr, outmap_ptr);
}
