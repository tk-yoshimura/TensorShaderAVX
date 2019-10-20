#include "../TensorShaderAvxBackend.h"

using namespace System;

void linear_zoom_1d(unsigned int channels,
                    unsigned int inwidth, unsigned int outwidth,
                    unsigned int th,
                    float* inmap_ptr, float* outmap_ptr) {

    const unsigned int inmap_offset = channels * inwidth * th, outmap_offset = channels * outwidth * th;
    inmap_ptr += inmap_offset;
    outmap_ptr += outmap_offset;

    const unsigned int j = channels & ~7u, k = channels - j;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);
    
    const __m256 cinv3 = _mm256_set1_ps(float(1.0 / 3.0));
    const __m256 c2 = _mm256_set1_ps(2);

    const unsigned int l = 0, r = channels;

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
}

void TensorShaderAvxBackend::Zoom::LinearZoom1D(unsigned int channels, unsigned int inwidth, unsigned int batch, unsigned int th, cli::array<float>^ inmap, cli::array<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, outmap);

    unsigned int outwidth = inwidth * 2;

    if (channels * inwidth * batch > (unsigned int)inmap->Length) {
        throw gcnew System::ArgumentException();
    }
    if (channels * outwidth * batch > (unsigned int)outmap->Length) {
        throw gcnew System::ArgumentException();
    }
    if (th >= batch) {
        throw gcnew System::ArgumentException();
    }

    pin_ptr<float> pinptr_inmap = &inmap[0];
    pin_ptr<float> pinptr_outmap = &outmap[0];

    float* inmap_ptr = pinptr_inmap;
    float* outmap_ptr = pinptr_outmap;

    linear_zoom_1d(channels, inwidth, outwidth, th, inmap_ptr, outmap_ptr);
}
