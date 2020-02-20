#include "../TensorShaderAvxBackend.h"

using namespace System;

void edge_padding_2d(unsigned int channels,
    unsigned int inwidth, unsigned int inheight,
    unsigned int outwidth, unsigned int outheight,
    unsigned int th,
    unsigned int pad_left, unsigned int pad_right,
    unsigned int pad_top, unsigned int pad_bottom,
    const float* __restrict inmap_ptr, float* outmap_ptr) {

    const unsigned int inmap_offset = channels * inwidth * inheight * th, outmap_offset = channels * outwidth * outheight * th;
    inmap_ptr += inmap_offset;
    outmap_ptr += outmap_offset;

    /* xy center */ {
        const unsigned int length = channels * inwidth;
        const unsigned int j = length & ~7u, k = length - j;
        const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        unsigned int inmap_idx = 0;
        unsigned int outmap_idx = channels * (pad_left + outwidth * pad_top);

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
        const unsigned int length = channels;
        const unsigned int j = length & ~7u, k = length - j;
        const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        unsigned int inmap_idx = channels * (pad_left + outwidth * pad_top);
        unsigned int outmap_org = channels * outwidth * pad_top;

        for (unsigned int iy = 0; iy < inheight; iy++) {
            unsigned int outmap_idx = outmap_org;

            for (unsigned int px = 0; px < pad_left; px++) {

                for (unsigned int i = 0; i < j; i += 8) {
                    __m256 x = _mm256_loadu_ps(outmap_ptr + inmap_idx + i);
                    _mm256_storeu_ps(outmap_ptr + outmap_idx + i, x);
                }
                if (k > 0) {
                    __m256 x = _mm256_maskload_ps(outmap_ptr + inmap_idx + j, mask);
                    _mm256_maskstore_ps(outmap_ptr + outmap_idx + j, mask, x);
                }

                outmap_idx += channels;
            }

            inmap_idx += channels * outwidth;
            outmap_org += channels * outwidth;
        }
    }

    /* x right */ {
        const unsigned int length = channels;
        const unsigned int j = length & ~7u, k = length - j;
        const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        unsigned int inmap_idx = channels * (pad_left + inwidth - 1 + outwidth * pad_top);
        unsigned int outmap_org = channels * (pad_left + inwidth + outwidth * pad_top);

        for (unsigned int iy = 0; iy < inheight; iy++) {
            unsigned int outmap_idx = outmap_org;

            for (unsigned int px = 0; px < pad_right; px++) {

                for (unsigned int i = 0; i < j; i += 8) {
                    __m256 x = _mm256_loadu_ps(outmap_ptr + inmap_idx + i);
                    _mm256_storeu_ps(outmap_ptr + outmap_idx + i, x);
                }
                if (k > 0) {
                    __m256 x = _mm256_maskload_ps(outmap_ptr + inmap_idx + j, mask);
                    _mm256_maskstore_ps(outmap_ptr + outmap_idx + j, mask, x);
                }

                outmap_idx += channels;
            }

            inmap_idx += channels * outwidth;
            outmap_org += channels * outwidth;
        }
    }

    /* y top */ {
        const unsigned int length = channels * outwidth;
        const unsigned int j = length & ~7u, k = length - j;
        const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        unsigned int inmap_idx = channels * outwidth * pad_top;
        unsigned int outmap_idx = 0;

        for (unsigned int py = 0; py < pad_top; py++) {
            for (unsigned int i = 0; i < j; i += 8) {
                __m256 x = _mm256_loadu_ps(outmap_ptr + inmap_idx + i);
                _mm256_storeu_ps(outmap_ptr + outmap_idx + i, x);
            }
            if (k > 0) {
                __m256 x = _mm256_maskload_ps(outmap_ptr + inmap_idx + j, mask);
                _mm256_maskstore_ps(outmap_ptr + outmap_idx + j, mask, x);
            }

            outmap_idx += channels * outwidth;
        }
    }

    /* y bottom */ {
        const unsigned int length = channels * outwidth;
        const unsigned int j = length & ~7u, k = length - j;
        const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

        unsigned int inmap_idx = channels * outwidth * (pad_top + inheight - 1);
        unsigned int outmap_idx = channels * outwidth * (pad_top + inheight);

        for (unsigned int py = 0; py < pad_bottom; py++) {
            for (unsigned int i = 0; i < j; i += 8) {
                __m256 x = _mm256_loadu_ps(outmap_ptr + inmap_idx + i);
                _mm256_storeu_ps(outmap_ptr + outmap_idx + i, x);
            }
            if (k > 0) {
                __m256 x = _mm256_maskload_ps(outmap_ptr + inmap_idx + j, mask);
                _mm256_maskstore_ps(outmap_ptr + outmap_idx + j, mask, x);
            }

            outmap_idx += channels * outwidth;
        }
    }
}

void TensorShaderAvxBackend::Padding::EdgePadding2D(unsigned int channels, unsigned int inwidth, unsigned int inheight, unsigned int batch, unsigned int th, unsigned int pad_left, unsigned int pad_right, unsigned int pad_top, unsigned int pad_bottom, AvxArray<float>^ inmap, AvxArray<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, outmap);

    if (th >= batch) {
        throw gcnew System::ArgumentException();
    }

    unsigned int outwidth = inwidth + pad_left + pad_right;
    unsigned int outheight = inheight + pad_top + pad_bottom;

    Util::CheckLength(channels * inwidth * inheight * batch, inmap);
    Util::CheckLength(channels * outwidth * outheight * batch, outmap);

    float* inmap_ptr = (float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());

    edge_padding_2d(channels, inwidth, inheight, outwidth, outheight, th, pad_left, pad_right, pad_top, pad_bottom, inmap_ptr, outmap_ptr);
}
