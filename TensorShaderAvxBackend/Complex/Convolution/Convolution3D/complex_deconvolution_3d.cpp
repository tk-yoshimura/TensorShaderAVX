#include "../../../TensorShaderAvxBackend.h"

using namespace System;

__forceinline __m256d _mm256_complexmul_pd(__m256d u, __m256d v) {
    __m256d uri = u;
    __m256d vrr = _mm256_permute4x64_pd(v, _MM_PERM_CCAA);
    __m256d uir = _mm256_permute4x64_pd(u, _MM_PERM_CDAB);
    __m256d vii = _mm256_permute4x64_pd(v, _MM_PERM_DDBB);

    return _mm256_fmaddsub_pd(uri, vrr, _mm256_mul_pd(uir, vii));
}

__forceinline __m256d _mm256_complexmulgrad_pd(__m256d u, __m256d v) {
    __m256d uri = u;
    __m256d vrr = _mm256_permute4x64_pd(v, _MM_PERM_CCAA);
    __m256d uir = _mm256_permute4x64_pd(u, _MM_PERM_CDAB);
    __m256d vii = _mm256_permute4x64_pd(v, _MM_PERM_DDBB);

    return _mm256_fmsubadd_pd(uri, vrr, _mm256_mul_pd(uir, vii));
}

__forceinline __m128 _mm256d_sumwise2(__m256d hi, __m256d lo) {
    __m256d u = _mm256_add_pd(lo, hi);
    __m128d w = _mm_add_pd(_mm256_extractf128_pd(u, 1), _mm256_castpd256_pd128(u));

    return _mm_cvtpd_ps(w);
}

void complex_deconvolution_3d(unsigned int inchannels, unsigned int outchannels, 
                              unsigned int inwidth, unsigned int outwidth, unsigned int kwidth,
                              unsigned int inheight, unsigned int outheight, unsigned int kheight,
                              unsigned int indepth, unsigned int outdepth, unsigned int kdepth,
                              unsigned int stride, unsigned int th, const float* __restrict inmap_ptr, float* __restrict outmap_ptr, const float* __restrict kernel_ptr) {

    const unsigned int inmap_offset = inchannels * inwidth * inheight * indepth * th;
    const unsigned int outmap_offset = outchannels * outwidth * outheight * outdepth * th;
    const unsigned int inch_sep = inchannels & ~7u, inch_rem = inchannels - inch_sep;
    const unsigned int kernelinchannels = inchannels / 2;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(inch_rem);
    const __m128i mask2 = TensorShaderAvxBackend::masktable_m128(2);
    const __m256i index = _mm256_setr_epi32(0, 1, outchannels, outchannels + 1, 
                                            outchannels * 2, outchannels * 2 + 1, outchannels * 3, outchannels * 3 + 1);

    inmap_ptr += inmap_offset;
    outmap_ptr += outmap_offset;

    for (unsigned int iz = 0; iz < indepth; iz++) {
        for (unsigned int iy = 0; iy < inheight; iy++) {
            for (unsigned int ix = 0; ix < inwidth; ix++) {
                for (unsigned int outch = 0; outch < outchannels; outch += 2) {
                    for (unsigned int kz = 0, oz = iz * stride; kz < kdepth; kz++, oz++) {
                        for (unsigned int ky = 0, oy = iy * stride; ky < kheight; ky++, oy++) {
                            for (unsigned int kx = 0, ox = ix * stride; kx < kwidth; kx++, ox++) {
                                __m256d uv_hi = _mm256_setzero_pd(), uv_lo = _mm256_setzero_pd();

                                for (unsigned int inch = 0; inch < inch_sep; inch += 8) {
                                    __m256 u = _mm256_loadu_ps(inmap_ptr + inch + inchannels * (ix + inwidth * (iy + inheight * iz)));
                                    __m256 v = _mm256_i32gather_ps(kernel_ptr + outch + outchannels * (inch / 2 + kernelinchannels * (kx + kwidth * (ky + kheight * kz))), index, 4);

                                    __m256d u_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(u, 1));
                                    __m256d u_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(u));

                                    __m256d v_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1));
                                    __m256d v_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(v));

                                    uv_hi = _mm256_add_pd(_mm256_complexmul_pd(u_hi, v_hi), uv_hi);
                                    uv_lo = _mm256_add_pd(_mm256_complexmul_pd(u_lo, v_lo), uv_lo);
                                }

                                if (inch_rem > 0) {
                                    __m256 u = _mm256_maskload_ps(inmap_ptr + inch_sep + inchannels * (ix + inwidth * (iy + inheight * iz)), mask);
                                    __m256 v = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), kernel_ptr + outch + outchannels * (inch_sep / 2 + kernelinchannels * (kx + kwidth * (ky + kheight * kz))),
                                        index, _mm256_castsi256_ps(mask), 4);

                                    __m256d u_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(u, 1));
                                    __m256d u_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(u));

                                    __m256d v_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1));
                                    __m256d v_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(v));

                                    uv_hi = _mm256_add_pd(_mm256_complexmul_pd(u_hi, v_hi), uv_hi);
                                    uv_lo = _mm256_add_pd(_mm256_complexmul_pd(u_lo, v_lo), uv_lo);
                                }

                                __m128 uv = _mm_add_ps(_mm256d_sumwise2(uv_hi, uv_lo), _mm_maskload_ps(outmap_ptr + outch + outchannels * (ox + outwidth * (oy + outheight * oz)), mask2));
                                _mm_maskstore_ps(outmap_ptr + outch + outchannels * (ox + outwidth * (oy + outheight * oz)), mask2, uv);
                            }
                        }
                    }
                }
            }
        }
    }
}

void complex_deconvolution_3d_grad(unsigned int inchannels, unsigned int outchannels,
                                   unsigned int inwidth, unsigned int outwidth, unsigned int kwidth,
                                   unsigned int inheight, unsigned int outheight, unsigned int kheight,
                                   unsigned int indepth, unsigned int outdepth, unsigned int kdepth,
                                   unsigned int stride, unsigned int th, const float* __restrict inmap_ptr, float* __restrict outmap_ptr, const float* __restrict kernel_ptr) {

    const unsigned int inmap_offset = inchannels * inwidth * inheight * indepth * th;
    const unsigned int outmap_offset = outchannels * outwidth * outheight * outdepth * th;
    const unsigned int inch_sep = inchannels & ~7u, inch_rem = inchannels - inch_sep;
    const unsigned int kernelinchannels = inchannels / 2;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(inch_rem);
    const __m128i mask2 = TensorShaderAvxBackend::masktable_m128(2);
    const __m256i index = _mm256_setr_epi32(0, 1, outchannels, outchannels + 1,
        outchannels * 2, outchannels * 2 + 1, outchannels * 3, outchannels * 3 + 1);

    inmap_ptr += inmap_offset;
    outmap_ptr += outmap_offset;

    for (unsigned int iz = 0; iz < indepth; iz++) {
        for (unsigned int iy = 0; iy < inheight; iy++) {
            for (unsigned int ix = 0; ix < inwidth; ix++) {
                for (unsigned int outch = 0; outch < outchannels; outch += 2) {
                    for (unsigned int kz = 0, oz = iz * stride; kz < kdepth; kz++, oz++) {
                        for (unsigned int ky = 0, oy = iy * stride; ky < kheight; ky++, oy++) {
                            for (unsigned int kx = 0, ox = ix * stride; kx < kwidth; kx++, ox++) {
                                __m256d uv_hi = _mm256_setzero_pd(), uv_lo = _mm256_setzero_pd();

                                for (unsigned int inch = 0; inch < inch_sep; inch += 8) {
                                    __m256 u = _mm256_loadu_ps(inmap_ptr + inch + inchannels * (ix + inwidth * (iy + inheight * iz)));
                                    __m256 v = _mm256_i32gather_ps(kernel_ptr + outch + outchannels * (inch / 2 + kernelinchannels * (kx + kwidth * (ky + kheight * kz))), index, 4);

                                    __m256d u_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(u, 1));
                                    __m256d u_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(u));

                                    __m256d v_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1));
                                    __m256d v_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(v));

                                    uv_hi = _mm256_add_pd(_mm256_complexmulgrad_pd(u_hi, v_hi), uv_hi);
                                    uv_lo = _mm256_add_pd(_mm256_complexmulgrad_pd(u_lo, v_lo), uv_lo);
                                }

                                if (inch_rem > 0) {
                                    __m256 u = _mm256_maskload_ps(inmap_ptr + inch_sep + inchannels * (ix + inwidth * (iy + inheight * iz)), mask);
                                    __m256 v = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), kernel_ptr + outch + outchannels * (inch_sep / 2 + kernelinchannels * (kx + kwidth * (ky + kheight * kz))),
                                        index, _mm256_castsi256_ps(mask), 4);

                                    __m256d u_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(u, 1));
                                    __m256d u_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(u));

                                    __m256d v_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1));
                                    __m256d v_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(v));

                                    uv_hi = _mm256_add_pd(_mm256_complexmulgrad_pd(u_hi, v_hi), uv_hi);
                                    uv_lo = _mm256_add_pd(_mm256_complexmulgrad_pd(u_lo, v_lo), uv_lo);
                                }

                                __m128 uv = _mm_add_ps(_mm256d_sumwise2(uv_hi, uv_lo), _mm_maskload_ps(outmap_ptr + outch + outchannels * (ox + outwidth * (oy + outheight * oz)), mask2));
                                _mm_maskstore_ps(outmap_ptr + outch + outchannels * (ox + outwidth * (oy + outheight * oz)), mask2, uv);
                            }
                        }
                    }
                }
            }
        }
    }
}

void TensorShaderAvxBackend::Complex::Deconvolution3D(unsigned int inchannels, unsigned int outchannels, unsigned int outwidth, unsigned int outheight, unsigned int outdepth,
	                                                  unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int kheight, unsigned int kdepth, unsigned int stride, bool gradmode,
                                                      AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, kernel, outmap);

    if (inchannels % 2 != 0 || outchannels % 2 != 0) {
        throw gcnew System::ArgumentException();
    }

    if (th >= batch) {
        throw gcnew System::ArgumentException();
    }

    unsigned int inwidth = outwidth + 1 - kwidth;
    unsigned int inheight = outheight + 1 - kheight;
    unsigned int indepth = outdepth + 1 - kdepth;

    Util::CheckLength(inchannels * inwidth * inheight * indepth * batch, inmap);
    Util::CheckLength(outchannels * outwidth * outheight * outdepth * batch, outmap);
    Util::CheckLength(inchannels * outchannels * kwidth * kheight * kdepth / 2, kernel);

    outmap->Zeroset(outchannels * outwidth * outheight * outdepth * th, outchannels * outwidth * outheight * outdepth);

    float* inmap_ptr = (float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());
    float* kernel_ptr = (float*)(kernel->Ptr.ToPointer());

    if (gradmode) {
        complex_deconvolution_3d_grad(inchannels, outchannels, 
                                      inwidth, outwidth, kwidth,
                                      inheight, outheight, kheight,
                                      indepth, outdepth, kdepth,
                                      stride, th, inmap_ptr, outmap_ptr, kernel_ptr);
    }
    else {
        complex_deconvolution_3d(inchannels, outchannels, 
                                 inwidth, outwidth, kwidth,
                                 inheight, outheight, kheight,
                                 indepth, outdepth, kdepth,
                                 stride, th, inmap_ptr, outmap_ptr, kernel_ptr);
    }
}
