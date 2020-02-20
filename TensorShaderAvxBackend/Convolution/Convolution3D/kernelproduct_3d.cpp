#include "../../TensorShaderAvxBackend.h"

using namespace System;

void kernelproduct_3d(unsigned int inchannels, unsigned int outchannels,
                      unsigned inwidth, unsigned outwidth, unsigned kwidth,
                      unsigned inheight, unsigned outheight, unsigned kheight,
                      unsigned indepth, unsigned outdepth, unsigned kdepth,
                      unsigned stride, unsigned batch, unsigned int outch,
                      const float* __restrict inmap_ptr, float* __restrict outmap_ptr, const float* __restrict kernel_ptr) {

    const unsigned int inch_sep = inchannels & ~7u, inch_rem = inchannels - inch_sep;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(inch_rem);
    
    for (unsigned int inch = 0; inch < inch_sep; inch += 8) {

        for (unsigned int kz = 0; kz < kdepth; kz++) {
            for (unsigned int ky = 0; ky < kheight; ky++) {
                for (unsigned int kx = 0; kx < kwidth; kx++) {
                    __m256d uv_hi = _mm256_setzero_pd(), uv_lo = _mm256_setzero_pd();

                    for (unsigned int th = 0; th < batch; th++) {
                        for (unsigned int oz = 0, iz = kz; oz < outdepth; oz++, iz += stride) {
                            for (unsigned int oy = 0, iy = ky; oy < outheight; oy++, iy += stride) {
                                for (unsigned int ox = 0, ix = kx; ox < outwidth; ox++, ix += stride) {

                                    __m256 u = _mm256_loadu_ps(inmap_ptr + inch + inchannels * (ix + inwidth * (iy + inheight * (iz + indepth * th))));
                                    __m256d v = _mm256_set1_pd(outmap_ptr[outch + outchannels * (ox + outwidth * (oy + outheight * (oz + outdepth * th)))]);

                                    __m256d u_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(u, 1));
                                    __m256d u_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(u));

                                    uv_hi = _mm256_fmadd_pd(u_hi, v, uv_hi);
                                    uv_lo = _mm256_fmadd_pd(u_lo, v, uv_lo);
                                }
                            }
                        }
                    }

                    _mm_storeu_ps(kernel_ptr + inch + inchannels * (outch + outchannels * (kx + kwidth * (ky + kheight * kz))), _mm256_cvtpd_ps(uv_lo));
                    _mm_storeu_ps(kernel_ptr + inch + inchannels * (outch + outchannels * (kx + kwidth * (ky + kheight * kz))) + 4, _mm256_cvtpd_ps(uv_hi));
                }
            }
        }
    }

    if (inch_rem > 0) {
        for (unsigned int kz = 0; kz < kdepth; kz++) {
            for (unsigned int ky = 0; ky < kheight; ky++) {
                for (unsigned int kx = 0; kx < kwidth; kx++) {
                    __m256d uv_hi = _mm256_setzero_pd(), uv_lo = _mm256_setzero_pd();

                    for (unsigned int th = 0; th < batch; th++) {
                        for (unsigned int oz = 0, iz = kz; oz < outdepth; oz++, iz += stride) {
                            for (unsigned int oy = 0, iy = ky; oy < outheight; oy++, iy += stride) {
                                for (unsigned int ox = 0, ix = kx; ox < outwidth; ox++, ix += stride) {
                                    __m256 u = _mm256_maskload_ps(inmap_ptr + inch_sep + inchannels * (ix + inwidth * (iy + inheight * (iz + indepth * th))), mask);
                                    __m256d v = _mm256_set1_pd(outmap_ptr[outch + outchannels * (ox + outwidth * (oy + outheight * (oz + outdepth * th)))]);

                                    __m256d u_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(u, 1));
                                    __m256d u_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(u));

                                    uv_hi = _mm256_fmadd_pd(u_hi, v, uv_hi);
                                    uv_lo = _mm256_fmadd_pd(u_lo, v, uv_lo);
                                }
                            }
                        }
                    }

                    if (inch_rem > 4) {
                        _mm_storeu_ps(kernel_ptr + inch_sep + inchannels * (outch + outchannels * (kx + kwidth * (ky + kheight * kz))), _mm256_cvtpd_ps(uv_lo));
                        _mm_maskstore_ps(kernel_ptr + inch_sep + inchannels * (outch + outchannels * (kx + kwidth * (ky + kheight * kz))) + 4, TensorShaderAvxBackend::masktable_m128(inch_rem - 4), _mm256_cvtpd_ps(uv_hi));
                    }
                    else if (inch_rem >= 4) {
                        _mm_storeu_ps(kernel_ptr + inch_sep + inchannels * (outch + outchannels * (kx + kwidth * (ky + kheight * kz))), _mm256_cvtpd_ps(uv_lo));
                    }
                    else {
                        _mm_maskstore_ps(kernel_ptr + inch_sep + inchannels * (outch + outchannels * (kx + kwidth * (ky + kheight * kz))), TensorShaderAvxBackend::masktable_m128(inch_rem), _mm256_cvtpd_ps(uv_lo));
                    }
                }
            }
        }
    }
}

void TensorShaderAvxBackend::Convolution::KernelProduct3D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth, unsigned int inheight, unsigned int indepth,
                                                          unsigned int batch, unsigned int outch, unsigned int kwidth, unsigned int kheight, unsigned int kdepth, unsigned int stride,
                                                          AvxArray<float>^ inmap, AvxArray<float>^ outmap, AvxArray<float>^ kernel) {

    Util::CheckDuplicateArray(inmap, kernel, outmap);

    if (outch >= outchannels) {
        throw gcnew System::ArgumentException();
    }

    unsigned int outwidth = (inwidth - kwidth) / stride + 1;
    unsigned int outheight = (inheight - kheight) / stride + 1;
    unsigned int outdepth = (indepth - kdepth) / stride + 1;

    Util::CheckLength(inchannels * inwidth * inheight * indepth * batch, inmap);
    Util::CheckLength(outchannels * outwidth * outheight * outdepth * batch, outmap);
    Util::CheckLength(inchannels * outchannels * kwidth * kheight * kdepth, kernel);

    float* inmap_ptr = (float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());
    float* kernel_ptr = (float*)(kernel->Ptr.ToPointer());

    kernelproduct_3d(inchannels, outchannels, 
                     inwidth, outwidth, kwidth, 
                     inheight, outheight, kheight,
                     indepth, outdepth, kdepth,
                     stride, batch, outch, 
                     inmap_ptr, outmap_ptr, kernel_ptr);
}
