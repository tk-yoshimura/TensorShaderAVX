#include "../TensorShaderAvxBackend.h"

using namespace System;

void linear_zoom_2d(unsigned int channels,
                    unsigned int inwidth, unsigned int inheight,
                    unsigned int outwidth, unsigned int outheight,
                    unsigned int th,
                    float* inmap_ptr, float* outmap_ptr) {

    const unsigned int inmap_offset = channels * inwidth * inheight * th, outmap_offset = channels * outwidth * outheight * th;
    inmap_ptr += inmap_offset;
    outmap_ptr += outmap_offset;

    const unsigned int j = channels & ~7u, k = channels - j;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

    const __m256 cinv9 = _mm256_set1_ps(float(1.0 / 9.0));
    const __m256 c4 = _mm256_set1_ps(4);
    const __m256 c2 = _mm256_set1_ps(2);

    const unsigned int lu = 0, ru = channels, ld = channels * outwidth, rd = channels * (1 + outwidth);

    for (unsigned int iyc = 0, oy = 0; iyc < inheight; iyc++, oy += 2) {
        unsigned int iyu = ((iyc > 0) ? iyc - 1 : 0);
        unsigned int iyd = ((iyc < inheight - 1) ? iyc + 1 : inheight - 1);

        for (unsigned int ixc = 0, ox = 0; ixc < inwidth; ixc++, ox += 2) {
            unsigned int ixl = ((ixc > 0) ? ixc - 1 : 0);
            unsigned int ixr = ((ixc < inwidth - 1) ? ixc + 1 : inwidth - 1);

            unsigned int inmap_c_idx  = channels * (ixc + inwidth * iyc);
            unsigned int inmap_l_idx  = channels * (ixl + inwidth * iyc);
            unsigned int inmap_r_idx  = channels * (ixr + inwidth * iyc);
            unsigned int inmap_u_idx  = channels * (ixc + inwidth * iyu);
            unsigned int inmap_d_idx  = channels * (ixc + inwidth * iyd);
            unsigned int inmap_lu_idx = channels * (ixl + inwidth * iyu);
            unsigned int inmap_ru_idx = channels * (ixr + inwidth * iyu);
            unsigned int inmap_ld_idx = channels * (ixl + inwidth * iyd);
            unsigned int inmap_rd_idx = channels * (ixr + inwidth * iyd);

            unsigned int outmap_idx = channels * (ox + outwidth * oy);

            for (unsigned int i = 0; i < j; i += 8) {
                __m256 xc  = _mm256_mul_ps(_mm256_loadu_ps(inmap_ptr + inmap_c_idx + i), c4);
                __m256 xl  = _mm256_mul_ps(_mm256_loadu_ps(inmap_ptr + inmap_l_idx + i), c2);
                __m256 xr  = _mm256_mul_ps(_mm256_loadu_ps(inmap_ptr + inmap_r_idx + i), c2);
                __m256 xu  = _mm256_mul_ps(_mm256_loadu_ps(inmap_ptr + inmap_u_idx + i), c2);
                __m256 xd  = _mm256_mul_ps(_mm256_loadu_ps(inmap_ptr + inmap_d_idx + i), c2);
                __m256 xlu = _mm256_loadu_ps(inmap_ptr + inmap_lu_idx + i);
                __m256 xru = _mm256_loadu_ps(inmap_ptr + inmap_ru_idx + i);
                __m256 xld = _mm256_loadu_ps(inmap_ptr + inmap_ld_idx + i);
                __m256 xrd = _mm256_loadu_ps(inmap_ptr + inmap_rd_idx + i);

                __m256 ylu = _mm256_mul_ps(cinv9, _mm256_add_ps(_mm256_add_ps(xc, xl), _mm256_add_ps(xu, xlu)));
                __m256 yru = _mm256_mul_ps(cinv9, _mm256_add_ps(_mm256_add_ps(xc, xr), _mm256_add_ps(xu, xru)));
                __m256 yld = _mm256_mul_ps(cinv9, _mm256_add_ps(_mm256_add_ps(xc, xl), _mm256_add_ps(xd, xld)));
                __m256 yrd = _mm256_mul_ps(cinv9, _mm256_add_ps(_mm256_add_ps(xc, xr), _mm256_add_ps(xd, xrd)));

                _mm256_storeu_ps(outmap_ptr + outmap_idx + lu + i, ylu);
                _mm256_storeu_ps(outmap_ptr + outmap_idx + ru + i, yru);
                _mm256_storeu_ps(outmap_ptr + outmap_idx + ld + i, yld);
                _mm256_storeu_ps(outmap_ptr + outmap_idx + rd + i, yrd);
            }
            if (k > 0) {
                __m256 xc =  _mm256_mul_ps(_mm256_maskload_ps(inmap_ptr + inmap_c_idx + j, mask), c4);
                __m256 xl =  _mm256_mul_ps(_mm256_maskload_ps(inmap_ptr + inmap_l_idx + j, mask), c2);
                __m256 xr =  _mm256_mul_ps(_mm256_maskload_ps(inmap_ptr + inmap_r_idx + j, mask), c2);
                __m256 xu =  _mm256_mul_ps(_mm256_maskload_ps(inmap_ptr + inmap_u_idx + j, mask), c2);
                __m256 xd =  _mm256_mul_ps(_mm256_maskload_ps(inmap_ptr + inmap_d_idx + j, mask), c2);
                __m256 xlu = _mm256_maskload_ps(inmap_ptr + inmap_lu_idx + j, mask);
                __m256 xru = _mm256_maskload_ps(inmap_ptr + inmap_ru_idx + j, mask);
                __m256 xld = _mm256_maskload_ps(inmap_ptr + inmap_ld_idx + j, mask);
                __m256 xrd = _mm256_maskload_ps(inmap_ptr + inmap_rd_idx + j, mask);
                
                __m256 ylu = _mm256_mul_ps(cinv9, _mm256_add_ps(_mm256_add_ps(xc, xl), _mm256_add_ps(xu, xlu)));
                __m256 yru = _mm256_mul_ps(cinv9, _mm256_add_ps(_mm256_add_ps(xc, xr), _mm256_add_ps(xu, xru)));
                __m256 yld = _mm256_mul_ps(cinv9, _mm256_add_ps(_mm256_add_ps(xc, xl), _mm256_add_ps(xd, xld)));
                __m256 yrd = _mm256_mul_ps(cinv9, _mm256_add_ps(_mm256_add_ps(xc, xr), _mm256_add_ps(xd, xrd)));

                _mm256_maskstore_ps(outmap_ptr + outmap_idx + lu + j, mask, ylu);
                _mm256_maskstore_ps(outmap_ptr + outmap_idx + ru + j, mask, yru);
                _mm256_maskstore_ps(outmap_ptr + outmap_idx + ld + j, mask, yld);
                _mm256_maskstore_ps(outmap_ptr + outmap_idx + rd + j, mask, yrd);
            }
        }
    }
}

void TensorShaderAvxBackend::Zoom::LinearZoom2D(unsigned int channels, unsigned int inwidth, unsigned int inheight, unsigned int batch, unsigned int th, AvxArray<float>^ inmap, AvxArray<float>^ outmap) {

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

    linear_zoom_2d(channels, inwidth, inheight, outwidth, outheight, th, inmap_ptr, outmap_ptr);
}
