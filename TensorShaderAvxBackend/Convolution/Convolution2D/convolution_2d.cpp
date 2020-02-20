#include "../../TensorShaderAvxBackend.h"

using namespace System;

__forceinline __m128 _mm256d_sum(__m256d hi, __m256d lo) {
    __m256d u = _mm256_hadd_pd(lo, hi);
    __m256d v = _mm256_hadd_pd(u, _mm256_setzero_pd());
    __m128d w = _mm_add_pd(_mm256_extractf128_pd(v, 1), _mm256_castpd256_pd128(v));

    return _mm_cvtpd_ps(w);
}

void convolution_2d(unsigned int inchannels, unsigned int outchannels, 
                    unsigned int inwidth, unsigned int outwidth, unsigned int kwidth,
                    unsigned int inheight, unsigned int outheight, unsigned int kheight,
                    unsigned stride, unsigned int th, 
                    const float* __restrict inmap_ptr, float* __restrict outmap_ptr, const float* __restrict kernel_ptr) {
        
    const unsigned int inmap_offset = inchannels * inwidth * inheight * th, outmap_offset = outchannels * outwidth * outheight * th;
    const unsigned int inch_sep = inchannels & ~7u, inch_rem = inchannels - inch_sep;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(inch_rem);
    const __m128i mask1 = TensorShaderAvxBackend::masktable_m128(1);

    inmap_ptr += inmap_offset;
    outmap_ptr += outmap_offset;

    for (unsigned int oy = 0; oy < outheight; oy++) {
        for (unsigned int ox = 0; ox < outwidth; ox++) {
            for (unsigned int outch = 0; outch < outchannels; outch++) {
                __m256d uv_hi = _mm256_setzero_pd(), uv_lo = _mm256_setzero_pd();

                for (unsigned int ky = 0, iy = oy * stride; ky < kheight; ky++, iy++) {
                    for (unsigned int kx = 0, ix = ox * stride; kx < kwidth; kx++, ix++) {
                        for (unsigned int inch = 0; inch < inch_sep; inch += 8) {
                            __m256 u = _mm256_loadu_ps(inmap_ptr + inch + inchannels * (ix + inwidth * iy));
                            __m256 v = _mm256_loadu_ps(kernel_ptr + inch + inchannels * (outch + outchannels * (kx + kwidth * ky)));

                            __m256d u_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(u, 1));
                            __m256d u_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(u));

                            __m256d v_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1));
                            __m256d v_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(v));

                            uv_hi = _mm256_fmadd_pd(u_hi, v_hi, uv_hi);
                            uv_lo = _mm256_fmadd_pd(u_lo, v_lo, uv_lo);
                        }

                        if (inch_rem > 0) {
                            __m256 u = _mm256_maskload_ps(inmap_ptr + inch_sep + inchannels * (ix + inwidth * iy), mask);
                            __m256 v = _mm256_maskload_ps(kernel_ptr + inch_sep + inchannels * (outch + outchannels * (kx + kwidth * ky)), mask);

                            __m256d u_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(u, 1));
                            __m256d u_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(u));

                            __m256d v_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1));
                            __m256d v_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(v));

                            uv_hi = _mm256_fmadd_pd(u_hi, v_hi, uv_hi);
                            uv_lo = _mm256_fmadd_pd(u_lo, v_lo, uv_lo);
                        }
                    }
                }

                _mm_maskstore_ps(outmap_ptr + outch + outchannels * (ox + outwidth * oy), mask1, _mm256d_sum(uv_hi, uv_lo));
            }
        }
    }
}

void TensorShaderAvxBackend::Convolution::Convolution2D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth, unsigned int inheight,
                                                        unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int kheight, unsigned int stride,
                                                        AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, kernel, outmap);

    if (th >= batch) {
        throw gcnew System::ArgumentException();
    }

    unsigned int outwidth = (inwidth - kwidth) / stride + 1;
    unsigned int outheight = (inheight - kheight) / stride + 1;

    Util::CheckLength(inchannels * inwidth * inheight * batch, inmap);
    Util::CheckLength(outchannels * outwidth * outheight * batch, outmap);
    Util::CheckLength(inchannels * outchannels * kwidth * kheight, kernel);

    float* inmap_ptr = (float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());
    float* kernel_ptr = (float*)(kernel->Ptr.ToPointer());

    convolution_2d(inchannels, outchannels, 
                   inwidth, outwidth, kwidth,
                   inheight, outheight, kheight,
                   stride, th, 
                   inmap_ptr, outmap_ptr, kernel_ptr);
}
