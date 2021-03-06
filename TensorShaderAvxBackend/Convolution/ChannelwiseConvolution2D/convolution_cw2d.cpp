#include "../../TensorShaderAvxBackend.h"

using namespace System;

void convolution_cw2d(unsigned int channels, 
                      unsigned int inwidth, unsigned int outwidth, unsigned int kwidth,
                      unsigned int inheight, unsigned int outheight, unsigned int kheight,
                      unsigned stride, unsigned int th, 
                      float* inmap_ptr, float* outmap_ptr, float* kernel_ptr) {
        
    const unsigned int inmap_offset = channels * inwidth * inheight * th, outmap_offset = channels * outwidth * outheight * th;
    const unsigned int ch_sep = channels & ~7u, ch_rem = channels - ch_sep;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(ch_rem);
    
    inmap_ptr += inmap_offset;
    outmap_ptr += outmap_offset;

    for (unsigned int oy = 0; oy < outheight; oy++) {
        for (unsigned int ox = 0; ox < outwidth; ox++) {
            for (unsigned int ch = 0; ch < ch_sep; ch += 8) {
                __m256d uv_hi = _mm256_setzero_pd(), uv_lo = _mm256_setzero_pd();

                for (unsigned int ky = 0, iy = oy * stride; ky < kheight; ky++, iy++) {
                    for (unsigned int kx = 0, ix = ox * stride; kx < kwidth; kx++, ix++) {
                        __m256 u = _mm256_loadu_ps(inmap_ptr + ch + channels * (ix + inwidth * iy));
                        __m256 v = _mm256_loadu_ps(kernel_ptr + ch + channels * (kx + kwidth * ky));

                        __m256d u_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(u, 1));
                        __m256d u_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(u));

                        __m256d v_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1));
                        __m256d v_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(v));

                        uv_hi = _mm256_fmadd_pd(u_hi, v_hi, uv_hi);
                        uv_lo = _mm256_fmadd_pd(u_lo, v_lo, uv_lo);
                    }
                }

                _mm_storeu_ps(outmap_ptr + ch + channels * (ox + outwidth * oy), _mm256_cvtpd_ps(uv_lo));
                _mm_storeu_ps(outmap_ptr + ch + channels * (ox + outwidth * oy) + 4, _mm256_cvtpd_ps(uv_hi));
            }

            if (ch_rem > 0){
                __m256d uv_hi = _mm256_setzero_pd(), uv_lo = _mm256_setzero_pd();

                for (unsigned int ky = 0, iy = oy * stride; ky < kheight; ky++, iy++) {
                    for (unsigned int kx = 0, ix = ox * stride; kx < kwidth; kx++, ix++) {
                        __m256 u = _mm256_maskload_ps(inmap_ptr + ch_sep + channels * (ix + inwidth * iy), mask);
                        __m256 v = _mm256_maskload_ps(kernel_ptr + ch_sep + channels * (kx + kwidth * ky), mask);

                        __m256d u_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(u, 1));
                        __m256d u_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(u));

                        __m256d v_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1));
                        __m256d v_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(v));

                        uv_hi = _mm256_fmadd_pd(u_hi, v_hi, uv_hi);
                        uv_lo = _mm256_fmadd_pd(u_lo, v_lo, uv_lo);
                    }
                }

                if (ch_rem > 4) {
                    _mm_storeu_ps(outmap_ptr + ch_sep + channels * (ox + outwidth * oy), _mm256_cvtpd_ps(uv_lo));
                    _mm_maskstore_ps(outmap_ptr + ch_sep + channels * (ox + outwidth * oy) + 4, TensorShaderAvxBackend::masktable_m128(ch_rem - 4), _mm256_cvtpd_ps(uv_hi));
                }
                else if (ch_rem >= 4) {
                    _mm_storeu_ps(outmap_ptr + ch_sep + channels * (ox + outwidth * oy), _mm256_cvtpd_ps(uv_lo));
                }
                else {
                    _mm_maskstore_ps(outmap_ptr + ch_sep + channels * (ox + outwidth * oy), TensorShaderAvxBackend::masktable_m128(ch_rem), _mm256_cvtpd_ps(uv_lo));
                }
            }
        }
    }
}

void TensorShaderAvxBackend::Convolution::ChannelwiseConvolution2D(unsigned int channels, unsigned int inwidth, unsigned int inheight,
                                                                   unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int kheight, unsigned int stride,
                                                                   AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, kernel, outmap);

    if (th >= batch) {
        throw gcnew System::ArgumentException();
    }

    unsigned int outwidth = (inwidth - kwidth) / stride + 1;
    unsigned int outheight = (inheight - kheight) / stride + 1;

    Util::CheckLength(channels * inwidth * inheight * batch, inmap);
    Util::CheckLength(channels * outwidth * outheight * batch, outmap);
    Util::CheckLength(channels * kwidth * kheight, kernel);

    float* inmap_ptr = (float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());
    float* kernel_ptr = (float*)(kernel->Ptr.ToPointer());

    convolution_cw2d(channels, 
                     inwidth, outwidth, kwidth,
                     inheight, outheight, kheight,
                     stride, th, 
                     inmap_ptr, outmap_ptr, kernel_ptr);
}
