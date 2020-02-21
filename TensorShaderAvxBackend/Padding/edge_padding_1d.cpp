#include "../TensorShaderAvxBackend.h"

using namespace System;

void edge_padding_1d(unsigned int channels,
                     unsigned int inwidth,
                     unsigned int outwidth,
                     unsigned int batch,
                     unsigned int pad_left, unsigned int pad_right,
                     const float* __restrict inmap_ptr, float* __restrict outmap_ptr) {

    for (unsigned int th = 0; th < batch; th++) {

        /* x center */ {
            const unsigned int length = channels * inwidth;
            const unsigned int j = length & ~7u, k = length - j;
            const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

            const unsigned int inmap_idx = 0;
            const unsigned int outmap_idx = channels * pad_left;

            for (unsigned int i = 0; i < j; i += 8) {
                __m256 x = _mm256_loadu_ps(inmap_ptr + inmap_idx + i);
                _mm256_storeu_ps(outmap_ptr + outmap_idx + i, x);
            }
            if (k > 0) {
                __m256 x = _mm256_maskload_ps(inmap_ptr + inmap_idx + j, mask);
                _mm256_maskstore_ps(outmap_ptr + outmap_idx + j, mask, x);
            }
        }

        /* x left */ {
            const unsigned int length = channels;
            const unsigned int j = length & ~7u, k = length - j;
            const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

            const unsigned int inmap_idx = channels * pad_left;

            for (unsigned px = 0; px < pad_left; px++) {
                const unsigned int outmap_idx = channels * px;

                for (unsigned int i = 0; i < j; i += 8) {
                    __m256 x = _mm256_loadu_ps(outmap_ptr + inmap_idx + i);
                    _mm256_storeu_ps(outmap_ptr + outmap_idx + i, x);
                }
                if (k > 0) {
                    __m256 x = _mm256_maskload_ps(outmap_ptr + inmap_idx + j, mask);
                    _mm256_maskstore_ps(outmap_ptr + outmap_idx + j, mask, x);
                }
            }
        }

        /* x right */ {
            const unsigned int length = channels;
            const unsigned int j = length & ~7u, k = length - j;
            const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

            const unsigned int inmap_idx = channels * (pad_left + inwidth - 1);

            for (unsigned px = 0; px < pad_right; px++) {
                const unsigned int outmap_idx = channels * (pad_left + inwidth + px);

                for (unsigned int i = 0; i < j; i += 8) {
                    __m256 x = _mm256_loadu_ps(outmap_ptr + inmap_idx + i);
                    _mm256_storeu_ps(outmap_ptr + outmap_idx + i, x);
                }
                if (k > 0) {
                    __m256 x = _mm256_maskload_ps(outmap_ptr + inmap_idx + j, mask);
                    _mm256_maskstore_ps(outmap_ptr + outmap_idx + j, mask, x);
                }
            }
        }

        inmap_ptr += channels * inwidth;
        outmap_ptr += channels * outwidth;
    }
}

void TensorShaderAvxBackend::Padding::EdgePadding1D(unsigned int channels, unsigned int inwidth, unsigned int batch, unsigned int pad_left, unsigned int pad_right, AvxArray<float>^ inmap, AvxArray<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, outmap);

    unsigned int outwidth = inwidth + pad_left + pad_right;

    Util::CheckLength(channels * inwidth * batch, inmap);
    Util::CheckLength(channels * outwidth * batch, outmap);

    const float* inmap_ptr = (const float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());

    edge_padding_1d(channels, 
                    inwidth, 
                    outwidth, 
                    batch, 
                    pad_left, pad_right, 
                    inmap_ptr, outmap_ptr);
}
