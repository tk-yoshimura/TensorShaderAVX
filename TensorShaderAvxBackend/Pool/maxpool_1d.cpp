#include "../TensorShaderAvxBackend.h"

using namespace System;

void maxpool_1d(unsigned int channels, unsigned int inwidth, unsigned int th, unsigned int stride, float* inmap_ptr, float* outmap_ptr) {
    const unsigned int outwidth = inwidth / stride;
    const unsigned int inmap_offset = channels * inwidth * th, outmap_offset = channels * outwidth * th;
    const unsigned int chsep = channels & ~7, chrem = channels - chsep; 

    __m256i mask = TensorShaderAvxBackend::masktable_m256(chrem);

    inmap_ptr += inmap_offset;
    outmap_ptr += outmap_offset;

    for (unsigned int ox = 0, ix = 0; ox < outwidth; ox++, ix += stride) {
        for (unsigned int ch = 0; ch < chsep; ch += 8) {
            __m256 xmax = _mm256_loadu_ps(inmap_ptr + ch + ix * channels);

            for (unsigned int kx = 1; kx < stride; kx++) {
                __m256 x = _mm256_loadu_ps(inmap_ptr + ch + (ix + kx) * channels);

                xmax = _mm256_max_ps(x, xmax);
            }

            _mm256_storeu_ps(outmap_ptr + ch + ox * channels, xmax);
        }
        if (chrem > 0) {
            __m256 xmax = _mm256_maskload_ps(inmap_ptr + chsep + ix * channels, mask);

            for (unsigned int kx = 1; kx < stride; kx++) {
                __m256 x = _mm256_maskload_ps(inmap_ptr + chsep + (ix + kx) * channels, mask);

                xmax = _mm256_max_ps(x, xmax);
            }

            _mm256_maskstore_ps(outmap_ptr + chsep + ox * channels, mask, xmax);
        }
    }
}

void TensorShaderAvxBackend::Pool::MaxPool1D(unsigned int channels, unsigned int inwidth, unsigned int batch, unsigned int th, unsigned int stride, cli::array<float>^ inmap, cli::array<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, outmap);

    if (channels * inwidth * batch > (unsigned int)inmap->Length) {
        throw gcnew System::ArgumentException();
    }
    if (channels * (inwidth / stride) * batch > (unsigned int)outmap->Length) {
        throw gcnew System::ArgumentException();
    }
    if (th >= batch) {
        throw gcnew System::ArgumentException();
    }

    pin_ptr<float> pinptr_inmap = &inmap[0];
    pin_ptr<float> pinptr_outmap = &outmap[0];

    float* inmap_ptr = pinptr_inmap;
    float* outmap_ptr = pinptr_outmap;

    maxpool_1d(channels, inwidth, th, stride, inmap_ptr, outmap_ptr);
}
