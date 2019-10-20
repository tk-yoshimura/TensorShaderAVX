#include "../../TensorShaderAvxBackend.h"

using namespace System;

void deconvolution_cw1d(unsigned int channels, 
                        unsigned inwidth, unsigned outwidth, unsigned kwidth, 
                        unsigned stride, unsigned int th, 
                        float* inmap_ptr, float* outmap_ptr, float* kernel_ptr) {
    
    const unsigned int inmap_offset = channels * inwidth * th, outmap_offset = channels * outwidth * th;
    const unsigned int ch_sep = channels & ~7u, ch_rem = channels - ch_sep;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(ch_rem);

    inmap_ptr += inmap_offset;
    outmap_ptr += outmap_offset;

    for (unsigned int ix = 0; ix < inwidth; ix++) {
        for (unsigned int ch = 0; ch < ch_sep; ch += 8) {
            __m256 u = _mm256_loadu_ps(inmap_ptr + ch + channels * ix);
            
            for (unsigned int kx = 0, ox = ix * stride; kx < kwidth; kx++, ox++) {
                __m256 v = _mm256_loadu_ps(kernel_ptr + ch + channels * kx);

                __m256 uv = _mm256_add_ps(_mm256_mul_ps(u, v), _mm256_loadu_ps(outmap_ptr + ch + channels * ox));

                _mm256_storeu_ps(outmap_ptr + ch + channels * ox, uv);
            }
        }

        if (ch_rem > 0) {
            __m256 u = _mm256_maskload_ps(inmap_ptr + ch_sep + channels * ix, mask);

            for (unsigned int kx = 0, ox = ix * stride; kx < kwidth; kx++, ox++) {
                __m256 v = _mm256_maskload_ps(kernel_ptr + ch_sep + channels * kx, mask);

                __m256 uv = _mm256_add_ps(_mm256_mul_ps(u, v), _mm256_maskload_ps(outmap_ptr + ch_sep + channels * ox, mask));

                _mm256_maskstore_ps(outmap_ptr + ch_sep + channels * ox, mask, uv);
            }
        }
    }
}

void TensorShaderAvxBackend::Convolution::ChannelwiseDeconvolution1D(unsigned int channels, unsigned int outwidth,
                                                                     unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int stride,
                                                                     cli::array<float>^ inmap, cli::array<float>^ kernel, cli::array<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, kernel, outmap);

    unsigned int inwidth = (outwidth - kwidth) / stride + 1;

    if (channels * inwidth * batch > (unsigned int)inmap->Length) {
        throw gcnew System::ArgumentException();
    }
    if (channels * outwidth * batch > (unsigned int)outmap->Length) {
        throw gcnew System::ArgumentException();
    }
    if (channels * kwidth > (unsigned int)kernel->Length) {
        throw gcnew System::ArgumentException();
    }
    if (th >= batch) {
        throw gcnew System::ArgumentException();
    }

    ArrayManipulation::Zeroset(channels * outwidth * th, channels * outwidth, outmap);

    pin_ptr<float> pinptr_inmap = &inmap[0];
    pin_ptr<float> pinptr_outmap = &outmap[0];
    pin_ptr<float> pinptr_kernel = &kernel[0];

    float* inmap_ptr = pinptr_inmap;
    float* outmap_ptr = pinptr_outmap;
    float* kernel_ptr = pinptr_kernel;

    deconvolution_cw1d(channels,
                       inwidth, outwidth, kwidth,
                       stride, th,
                       inmap_ptr, outmap_ptr, kernel_ptr);
}
