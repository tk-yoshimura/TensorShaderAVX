#include "../../../TensorShaderAvxBackend.h"

using namespace System;

__forceinline __m256d _mm256_complexmulkernelgrad_pd(__m256d u, __m256d v) {
    __m256d vri = v;
    __m256d urr = _mm256_permute4x64_pd(u, _MM_PERM_CCAA);
    __m256d vir = _mm256_permute4x64_pd(v, _MM_PERM_CDAB);
    __m256d uii = _mm256_permute4x64_pd(u, _MM_PERM_DDBB);

    return _mm256_fmsubadd_pd(vri, urr, _mm256_mul_pd(vir, uii));
}

void complex_kernelproduct_2d(unsigned int inchannels, unsigned int outchannels, 
                              unsigned int inwidth, unsigned int outwidth, unsigned int kwidth, 
                              unsigned int inheight, unsigned int outheight, unsigned int kheight,
                              unsigned int batch, 
                              const float* __restrict inmap_ptr, const float* __restrict outmap_ptr, float* __restrict kernel_ptr) {

    const unsigned int inch_sep = inchannels & ~7u, inch_rem = inchannels - inch_sep;
    const unsigned int kerneloutchannels = outchannels / 2;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(inch_rem);

    for (unsigned int outch = 0, koutch = 0; outch < outchannels; outch += 2, koutch++) {
        for (unsigned int inch = 0; inch < inch_sep; inch += 8) {
            for (unsigned int ky = 0; ky < kheight; ky++) {
                for (unsigned int kx = 0; kx < kwidth; kx++) {
                    __m256d uv_hi = _mm256_setzero_pd(), uv_lo = _mm256_setzero_pd();

                    for (unsigned int th = 0; th < batch; th++) {
                        for (unsigned int oy = 0, iy = ky; oy < outheight; oy++, iy++) {
                            for (unsigned int ox = 0, ix = kx; ox < outwidth; ox++, ix++) {
                                __m256 u = _mm256_loadu_ps(inmap_ptr + inch + inchannels * (ix + inwidth * (iy + inheight * th)));

                                float vr = outmap_ptr[outch + outchannels * (ox + outwidth * (oy + outheight * th))];
                                float vi = outmap_ptr[outch + 1 + outchannels * (ox + outwidth * (oy + outheight * th))];
                                __m256d v = _mm256_setr_pd(vr, vi, vr, vi);

                                __m256d u_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(u, 1));
                                __m256d u_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(u));

                                uv_hi = _mm256_add_pd(_mm256_complexmulkernelgrad_pd(u_hi, v), uv_hi);
                                uv_lo = _mm256_add_pd(_mm256_complexmulkernelgrad_pd(u_lo, v), uv_lo);
                            }
                        }
                    }

                    _mm_storeu_ps(kernel_ptr + inch + inchannels * (koutch + kerneloutchannels * (kx + kwidth * ky)), _mm256_cvtpd_ps(uv_lo));
                    _mm_storeu_ps(kernel_ptr + inch + inchannels * (koutch + kerneloutchannels * (kx + kwidth * ky)) + 4, _mm256_cvtpd_ps(uv_hi));
                }
            }
        }

        if (inch_rem > 0) {
            for (unsigned int ky = 0; ky < kheight; ky++) {
                for (unsigned int kx = 0; kx < kwidth; kx++) {
                    __m256d uv_hi = _mm256_setzero_pd(), uv_lo = _mm256_setzero_pd();

                    for (unsigned int th = 0; th < batch; th++) {
                        for (unsigned int oy = 0, iy = ky; oy < outheight; oy++, iy++) {
                            for (unsigned int ox = 0, ix = kx; ox < outwidth; ox++, ix++) {
                                __m256 u = _mm256_maskload_ps(inmap_ptr + inch_sep + inchannels * (ix + inwidth * (iy + inheight * th)), mask);

                                float vr = outmap_ptr[outch + outchannels * (ox + outwidth * (oy + outheight * th))];
                                float vi = outmap_ptr[outch + 1 + outchannels * (ox + outwidth * (oy + outheight * th))];
                                __m256d v = _mm256_setr_pd(vr, vi, vr, vi);

                                __m256d u_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(u, 1));
                                __m256d u_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(u));

                                uv_hi = _mm256_add_pd(_mm256_complexmulkernelgrad_pd(u_hi, v), uv_hi);
                                uv_lo = _mm256_add_pd(_mm256_complexmulkernelgrad_pd(u_lo, v), uv_lo);
                            }
                        }
                    }

                    if (inch_rem > 4) {
                        _mm_storeu_ps(kernel_ptr + inch_sep + inchannels * (koutch + kerneloutchannels * (kx + kwidth * ky)), _mm256_cvtpd_ps(uv_lo));
                        _mm_maskstore_ps(kernel_ptr + inch_sep + inchannels * (koutch + kerneloutchannels * (kx + kwidth * ky)) + 4, TensorShaderAvxBackend::masktable_m128(inch_rem - 4), _mm256_cvtpd_ps(uv_hi));
                    }
                    else if (inch_rem >= 4) {
                        _mm_storeu_ps(kernel_ptr + inch_sep + inchannels * (koutch + kerneloutchannels * (kx + kwidth * ky)), _mm256_cvtpd_ps(uv_lo));
                    }
                    else {
                        _mm_maskstore_ps(kernel_ptr + inch_sep + inchannels * (koutch + kerneloutchannels * (kx + kwidth * ky)), TensorShaderAvxBackend::masktable_m128(inch_rem), _mm256_cvtpd_ps(uv_lo));
                    }
                }
            }
        }
    }
}

void complex_kernelproduct_2d_transpose(unsigned int inchannels, unsigned int outchannels,
                                        unsigned int inwidth, unsigned int outwidth, unsigned int kwidth,
                                        unsigned int inheight, unsigned int outheight, unsigned int kheight,
                                        unsigned int batch, 
                                        const float* __restrict inmap_ptr, const float* __restrict outmap_ptr, float* __restrict kernel_ptr) {

    const unsigned int inch_sep = inchannels & ~7u, inch_rem = inchannels - inch_sep;
    const unsigned int kerneloutchannels = outchannels / 2;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(inch_rem);

    for (unsigned int outch = 0, koutch = 0; outch < outchannels; outch += 2, koutch++) {
        for (unsigned int inch = 0; inch < inch_sep; inch += 8) {
            for (unsigned int ky = 0; ky < kheight; ky++) {
                for (unsigned int kx = 0; kx < kwidth; kx++) {
                    __m256d vu_hi = _mm256_setzero_pd(), vu_lo = _mm256_setzero_pd();

                    for (unsigned int th = 0; th < batch; th++) {
                        for (unsigned int oy = 0, iy = ky; oy < outheight; oy++, iy++) {
                            for (unsigned int ox = 0, ix = kx; ox < outwidth; ox++, ix++) {
                                __m256 u = _mm256_loadu_ps(inmap_ptr + inch + inchannels * (ix + inwidth * (iy + inheight * th)));

                                float vr = outmap_ptr[outch + outchannels * (ox + outwidth * (oy + outheight * th))];
                                float vi = outmap_ptr[outch + 1 + outchannels * (ox + outwidth * (oy + outheight * th))];
                                __m256d v = _mm256_setr_pd(vr, vi, vr, vi);

                                __m256d u_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(u, 1));
                                __m256d u_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(u));

                                vu_hi = _mm256_add_pd(_mm256_complexmulkernelgrad_pd(v, u_hi), vu_hi);
                                vu_lo = _mm256_add_pd(_mm256_complexmulkernelgrad_pd(v, u_lo), vu_lo);
                            }
                        }
                    }

                    _mm_storeu_ps(kernel_ptr + inch + inchannels * (koutch + kerneloutchannels * (kx + kwidth * ky)), _mm256_cvtpd_ps(vu_lo));
                    _mm_storeu_ps(kernel_ptr + inch + inchannels * (koutch + kerneloutchannels * (kx + kwidth * ky)) + 4, _mm256_cvtpd_ps(vu_hi));
                }
            }
        }

        if (inch_rem > 0) {
            for (unsigned int ky = 0; ky < kheight; ky++) {
                for (unsigned int kx = 0; kx < kwidth; kx++) {
                    __m256d vu_hi = _mm256_setzero_pd(), vu_lo = _mm256_setzero_pd();

                    for (unsigned int th = 0; th < batch; th++) {
                        for (unsigned int oy = 0, iy = ky; oy < outheight; oy++, iy++) {
                            for (unsigned int ox = 0, ix = kx; ox < outwidth; ox++, ix++) {
                                __m256 u = _mm256_maskload_ps(inmap_ptr + inch_sep + inchannels * (ix + inwidth * (iy + inheight * th)), mask);

                                float vr = outmap_ptr[outch + outchannels * (ox + outwidth * (oy + outheight * th))];
                                float vi = outmap_ptr[outch + 1 + outchannels * (ox + outwidth * (oy + outheight * th))];
                                __m256d v = _mm256_setr_pd(vr, vi, vr, vi);

                                __m256d u_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(u, 1));
                                __m256d u_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(u));

                                vu_hi = _mm256_add_pd(_mm256_complexmulkernelgrad_pd(v, u_hi), vu_hi);
                                vu_lo = _mm256_add_pd(_mm256_complexmulkernelgrad_pd(v, u_lo), vu_lo);
                            }
                        }
                    }

                    if (inch_rem > 4) {
                        _mm_storeu_ps(kernel_ptr + inch_sep + inchannels * (koutch + kerneloutchannels * (kx + kwidth * ky)), _mm256_cvtpd_ps(vu_lo));
                        _mm_maskstore_ps(kernel_ptr + inch_sep + inchannels * (koutch + kerneloutchannels * (kx + kwidth * ky)) + 4, TensorShaderAvxBackend::masktable_m128(inch_rem - 4), _mm256_cvtpd_ps(vu_hi));
                    }
                    else if (inch_rem >= 4) {
                        _mm_storeu_ps(kernel_ptr + inch_sep + inchannels * (koutch + kerneloutchannels * (kx + kwidth * ky)), _mm256_cvtpd_ps(vu_lo));
                    }
                    else {
                        _mm_maskstore_ps(kernel_ptr + inch_sep + inchannels * (koutch + kerneloutchannels * (kx + kwidth * ky)), TensorShaderAvxBackend::masktable_m128(inch_rem), _mm256_cvtpd_ps(vu_lo));
                    }
                }
            }
        }
    }
}

void TensorShaderAvxBackend::Complex::KernelProduct2D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth, unsigned int inheight, 
	                                                  unsigned int batch, unsigned int kwidth, unsigned int kheight, bool transpose,
                                                      AvxArray<float>^ inmap, AvxArray<float>^ outmap, AvxArray<float>^ kernel) {

    Util::CheckDuplicateArray(inmap, kernel, outmap);

    if (inchannels % 2 != 0 || outchannels % 2 != 0) {
        throw gcnew System::ArgumentException();
    }

    unsigned int outwidth = inwidth + 1 - kwidth;
    unsigned int outheight = inheight + 1 - kheight;

    Util::CheckLength(inchannels * inwidth * inheight * batch, inmap);
    Util::CheckLength(outchannels * outwidth * outheight * batch, outmap);
    Util::CheckLength(inchannels * outchannels * kwidth * kheight / 2, kernel);

    const float* inmap_ptr = (const float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());
    float* kernel_ptr = (float*)(kernel->Ptr.ToPointer());

    if (transpose) {
        complex_kernelproduct_2d_transpose(inchannels, outchannels, 
                                           inwidth, outwidth, kwidth,
                                           inheight, outheight, kheight,
                                           batch, 
                                           inmap_ptr, outmap_ptr, kernel_ptr);
    }
    else {
        complex_kernelproduct_2d(inchannels, outchannels, 
                                 inwidth, outwidth, kwidth,
                                 inheight, outheight, kheight,
                                 batch, 
                                 inmap_ptr, outmap_ptr, kernel_ptr);
    }
}
