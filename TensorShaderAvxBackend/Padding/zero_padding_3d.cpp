#include "../TensorShaderAvxBackend.h"

using namespace System;

void zero_padding_3d(unsigned int channels,
                     unsigned int inwidth, unsigned int inheight, unsigned int indepth,
                     unsigned int outwidth, unsigned int outheight, unsigned int outdepth,
                     unsigned int batch,
                     unsigned int pad_left, unsigned int pad_right,
                     unsigned int pad_top, unsigned int pad_bottom,
                     unsigned int pad_front, unsigned int pad_rear,
                     const float* __restrict inmap_ptr, float* __restrict outmap_ptr) {

    for (unsigned int th = 0; th < batch; th++) {

        for (unsigned int iz = 0; iz < indepth; iz++) {
            /* xyz center */ {
                const unsigned int length = channels * inwidth;
                const unsigned int j = length & ~7u, k = length - j;
                const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

                unsigned int inmap_idx = channels * inwidth * inheight * iz;
                unsigned int outmap_idx = channels * (pad_left + outwidth * (pad_top + outheight * (iz + pad_front)));

                for (unsigned int iy = 0; iy < inheight; iy++) {
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

            /* x left */ {
                const unsigned int length = channels * pad_left;
                const unsigned int j = length & ~7u, k = length - j;
                const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);
                const __m256 x = _mm256_setzero_ps();

                unsigned int outmap_idx = channels * outwidth * (pad_top + outheight * (iz + pad_front));

                for (unsigned int iy = 0; iy < inheight; iy++) {
                    for (unsigned int i = 0; i < j; i += 8) {
                        _mm256_storeu_ps(outmap_ptr + outmap_idx + i, x);
                    }
                    if (k > 0) {
                        _mm256_maskstore_ps(outmap_ptr + outmap_idx + j, mask, x);
                    }

                    outmap_idx += channels * outwidth;
                }
            }

            /* x right */ {
                const unsigned int length = channels * pad_right;
                const unsigned int j = length & ~7u, k = length - j;
                const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);
                const __m256 x = _mm256_setzero_ps();

                unsigned int outmap_idx = channels * (pad_left + inwidth + outwidth * (pad_top + outheight * (iz + pad_front)));

                for (unsigned int iy = 0; iy < inheight; iy++) {
                    for (unsigned int i = 0; i < j; i += 8) {
                        _mm256_storeu_ps(outmap_ptr + outmap_idx + i, x);
                    }
                    if (k > 0) {
                        _mm256_maskstore_ps(outmap_ptr + outmap_idx + j, mask, x);
                    }

                    outmap_idx += channels * outwidth;
                }
            }

            /* y top */ {
                const unsigned int length = channels * outwidth * pad_top;
                const unsigned int j = length & ~7u, k = length - j;
                const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);
                const __m256 x = _mm256_setzero_ps();

                const unsigned int outmap_idx = channels * outwidth * outheight * (iz + pad_front);

                for (unsigned int i = 0; i < j; i += 8) {
                    _mm256_storeu_ps(outmap_ptr + outmap_idx + i, x);
                }
                if (k > 0) {
                    _mm256_maskstore_ps(outmap_ptr + outmap_idx + j, mask, x);
                }
            }

            /* y bottom */ {
                const unsigned int length = channels * outwidth * pad_bottom;
                const unsigned int j = length & ~7u, k = length - j;
                const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);
                const __m256 x = _mm256_setzero_ps();

                const unsigned int outmap_idx = channels * outwidth * (pad_top + inheight + outheight * (iz + pad_front));

                for (unsigned int i = 0; i < j; i += 8) {
                    _mm256_storeu_ps(outmap_ptr + outmap_idx + i, x);
                }
                if (k > 0) {
                    _mm256_maskstore_ps(outmap_ptr + outmap_idx + j, mask, x);
                }
            }
        }

        /*z front*/ {
            const unsigned int length = channels * outwidth * outheight * pad_front;
            const unsigned int j = length & ~7u, k = length - j;
            const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);
            const __m256 x = _mm256_setzero_ps();

            const unsigned int outmap_idx = 0;

            for (unsigned int i = 0; i < j; i += 8) {
                _mm256_storeu_ps(outmap_ptr + outmap_idx + i, x);
            }
            if (k > 0) {
                _mm256_maskstore_ps(outmap_ptr + outmap_idx + j, mask, x);
            }
        }

        /*z rear*/ {
            const unsigned int length = channels * outwidth * outheight * pad_rear;
            const unsigned int j = length & ~7u, k = length - j;
            const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);
            const __m256 x = _mm256_setzero_ps();

            const unsigned int outmap_idx = channels * outwidth * outheight * (pad_front + indepth);

            for (unsigned int i = 0; i < j; i += 8) {
                _mm256_storeu_ps(outmap_ptr + outmap_idx + i, x);
            }
            if (k > 0) {
                _mm256_maskstore_ps(outmap_ptr + outmap_idx + j, mask, x);
            }
        }

        inmap_ptr += channels * inwidth * inheight * indepth;
        outmap_ptr += channels * outwidth * outheight * outdepth;
    }
}

void TensorShaderAvxBackend::Padding::ZeroPadding3D(unsigned int channels, unsigned int inwidth, unsigned int inheight, unsigned int indepth, unsigned int batch, unsigned int pad_left, unsigned int pad_right, unsigned int pad_top, unsigned int pad_bottom, unsigned int pad_front, unsigned int pad_rear, AvxArray<float>^ inmap, AvxArray<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, outmap);

    unsigned int outwidth = inwidth + pad_left + pad_right;
    unsigned int outheight = inheight + pad_top + pad_bottom;
    unsigned int outdepth = indepth + pad_front + pad_rear;

    Util::CheckLength(channels * inwidth * inheight * indepth * batch, inmap);
    Util::CheckLength(channels * outwidth * outheight * outdepth * batch, outmap);

    const float* inmap_ptr = (const float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());

    zero_padding_3d(channels, inwidth, inheight, indepth, outwidth, outheight, outdepth, batch, pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_rear, inmap_ptr, outmap_ptr);
}
