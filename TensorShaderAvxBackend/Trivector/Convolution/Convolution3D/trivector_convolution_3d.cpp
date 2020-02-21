#include "../../../TensorShaderAvxBackend.h"

using namespace System;

__forceinline __m256d _mm256_trivectormul_pd(__m256d v, __m256d q) {
    const __m256d sign_pmm = _mm256_castsi256_pd(_mm256_setr_epi64x(0, 0x8000000000000000ull, 0x8000000000000000ull, 0));
    const __m256d sign_mpm = _mm256_castsi256_pd(_mm256_setr_epi64x(0x8000000000000000ull, 0, 0x8000000000000000ull, 0));
    const __m256d sign_mmp = _mm256_castsi256_pd(_mm256_setr_epi64x(0x8000000000000000ull, 0x8000000000000000ull, 0, 0));

    __m256d s = _mm256_mul_pd(q, q);
    __m256d sx = _mm256_permute4x64_pd(s, _MM_PERM_AAAA);
    __m256d sy = _mm256_xor_pd(sign_pmm, _mm256_permute4x64_pd(s, _MM_PERM_BBBB));
    __m256d sz = _mm256_xor_pd(sign_mpm, _mm256_permute4x64_pd(s, _MM_PERM_CCCC));
    __m256d sw = _mm256_xor_pd(sign_mmp, _mm256_permute4x64_pd(s, _MM_PERM_DDDD));

    __m256d ss = _mm256_add_pd(_mm256_add_pd(sx, sy), _mm256_add_pd(sz, sw));

    __m256d q_yzw = _mm256_permute4x64_pd(q, _MM_PERM_ADCB), q_zwy = _mm256_permute4x64_pd(q, _MM_PERM_ABDC), q_xxx = _mm256_permute4x64_pd(q, _MM_PERM_AAAA);
    __m256d v_xyz = v, v_yzx = _mm256_permute4x64_pd(v, _MM_PERM_DACB), v_zxy = _mm256_permute4x64_pd(v, _MM_PERM_DBAC);

    __m256d m_xyz = _mm256_mul_pd(q_yzw, q_zwy), m_zxy = _mm256_permute4x64_pd(m_xyz, _MM_PERM_DBAC);
    __m256d n_xyz = _mm256_mul_pd(q_xxx, q_yzw), n_zxy = _mm256_permute4x64_pd(n_xyz, _MM_PERM_DBAC), n_yzx = _mm256_permute4x64_pd(n_xyz, _MM_PERM_DACB);

    __m256d vmmn = _mm256_mul_pd(v_yzx, _mm256_sub_pd(m_xyz, n_zxy));
    __m256d vpmn = _mm256_mul_pd(v_zxy, _mm256_add_pd(m_zxy, n_yzx));
    __m256d vmn = _mm256_add_pd(vmmn, vpmn);
    __m256d vmnx2 = _mm256_add_pd(vmn, vmn);

    __m256d u = _mm256_fmadd_pd(v_xyz, ss, vmnx2);

    return u;
}

__forceinline __m256d _mm256_trivectormulgrad_pd(__m256d v, __m256d q) {
    const __m256d sign_pmm = _mm256_castsi256_pd(_mm256_setr_epi64x(0, 0x8000000000000000ull, 0x8000000000000000ull, 0));
    const __m256d sign_mpm = _mm256_castsi256_pd(_mm256_setr_epi64x(0x8000000000000000ull, 0, 0x8000000000000000ull, 0));
    const __m256d sign_mmp = _mm256_castsi256_pd(_mm256_setr_epi64x(0x8000000000000000ull, 0x8000000000000000ull, 0, 0));

    __m256d s = _mm256_mul_pd(q, q);
    __m256d sx = _mm256_permute4x64_pd(s, _MM_PERM_AAAA);
    __m256d sy = _mm256_xor_pd(sign_pmm, _mm256_permute4x64_pd(s, _MM_PERM_BBBB));
    __m256d sz = _mm256_xor_pd(sign_mpm, _mm256_permute4x64_pd(s, _MM_PERM_CCCC));
    __m256d sw = _mm256_xor_pd(sign_mmp, _mm256_permute4x64_pd(s, _MM_PERM_DDDD));

    __m256d ss = _mm256_add_pd(_mm256_add_pd(sx, sy), _mm256_add_pd(sz, sw));

    __m256d q_yzw = _mm256_permute4x64_pd(q, _MM_PERM_ADCB), q_zwy = _mm256_permute4x64_pd(q, _MM_PERM_ABDC), q_xxx = _mm256_permute4x64_pd(q, _MM_PERM_AAAA);
    __m256d v_xyz = v, v_yzx = _mm256_permute4x64_pd(v, _MM_PERM_DACB), v_zxy = _mm256_permute4x64_pd(v, _MM_PERM_DBAC);

    __m256d m_xyz = _mm256_mul_pd(q_yzw, q_zwy), m_zxy = _mm256_permute4x64_pd(m_xyz, _MM_PERM_DBAC);
    __m256d n_xyz = _mm256_mul_pd(q_xxx, q_yzw), n_zxy = _mm256_permute4x64_pd(n_xyz, _MM_PERM_DBAC), n_yzx = _mm256_permute4x64_pd(n_xyz, _MM_PERM_DACB);

    __m256d vmmn = _mm256_mul_pd(v_yzx, _mm256_add_pd(m_xyz, n_zxy));
    __m256d vpmn = _mm256_mul_pd(v_zxy, _mm256_sub_pd(m_zxy, n_yzx));
    __m256d vmn = _mm256_add_pd(vmmn, vpmn);
    __m256d vmnx2 = _mm256_add_pd(vmn, vmn);

    __m256d u = _mm256_fmadd_pd(v_xyz, ss, vmnx2);

    return u;
}

void trivector_convolution_3d(unsigned int inchannels, unsigned int outchannels, 
                              unsigned int inwidth, unsigned int outwidth, unsigned int kwidth,
                              unsigned int inheight, unsigned int outheight, unsigned int kheight,
                              unsigned int indepth, unsigned int outdepth, unsigned int kdepth,
                              unsigned int stride, unsigned int th, const float* __restrict inmap_ptr, float* __restrict outmap_ptr, const float* __restrict kernel_ptr) {

    const unsigned int inmap_offset = inchannels * inwidth * inheight * indepth * th;
    const unsigned int outmap_offset = outchannels * outwidth * outheight * outdepth * th;
    const unsigned int kernelinchannels = inchannels / 3 * 4, kerneloutchannels = outchannels / 3;
    const __m128i mask3 = TensorShaderAvxBackend::masktable_m128(3);

    inmap_ptr += inmap_offset;
    outmap_ptr += outmap_offset;

    for (unsigned int oz = 0; oz < outdepth; oz++) {
        for (unsigned int oy = 0; oy < outheight; oy++) {
            for (unsigned int ox = 0; ox < outwidth; ox++) {
                for (unsigned int outch = 0, koutch = 0; outch < outchannels; outch += 3, koutch++) {
                    __m256d vq = _mm256_setzero_pd();

                    for (unsigned int kz = 0, iz = oz * stride; kz < kdepth; kz++, iz++) {
                        for (unsigned int ky = 0, iy = oy * stride; ky < kheight; ky++, iy++) {
                            for (unsigned int kx = 0, ix = ox * stride; kx < kwidth; kx++, ix++) {
                                for (unsigned int inch = 0, kinch = 0; inch < inchannels; inch += 3, kinch += 4) {
                                    __m256d v = _mm256_cvtps_pd(_mm_maskload_ps(inmap_ptr + inch + inchannels * (ix + inwidth * (iy + inheight * iz)), mask3));
                                    __m256d q = _mm256_cvtps_pd(_mm_load_ps(kernel_ptr + kinch + kernelinchannels * (koutch + kerneloutchannels * (kx + kwidth * (ky + kheight * kz)))));

                                    vq = _mm256_add_pd(_mm256_trivectormul_pd(v, q), vq);
                                }
                            }
                        }
                    }

                    _mm_maskstore_ps(outmap_ptr + outch + outchannels * (ox + outwidth * (oy + outheight * oz)), mask3, _mm256_cvtpd_ps(vq));
                }
            }
        }
    }
}

void trivector_convolution_3d_grad(unsigned int inchannels, unsigned int outchannels,
                                   unsigned int inwidth, unsigned int outwidth, unsigned int kwidth,
                                   unsigned int inheight, unsigned int outheight, unsigned int kheight,
                                   unsigned int indepth, unsigned int outdepth, unsigned int kdepth,
                                   unsigned int stride, unsigned int th, const float* __restrict inmap_ptr, float* __restrict outmap_ptr, const float* __restrict kernel_ptr) {

    const unsigned int inmap_offset = inchannels * inwidth * inheight * indepth * th;
    const unsigned int outmap_offset = outchannels * outwidth * outheight * outdepth * th;
    const unsigned int kernelinchannels = inchannels / 3 * 4, kerneloutchannels = outchannels / 3;
    const __m128i mask3 = TensorShaderAvxBackend::masktable_m128(3);

    inmap_ptr += inmap_offset;
    outmap_ptr += outmap_offset;

    for (unsigned int oz = 0; oz < outdepth; oz++) {
        for (unsigned int oy = 0; oy < outheight; oy++) {
            for (unsigned int ox = 0; ox < outwidth; ox++) {
                for (unsigned int outch = 0, koutch = 0; outch < outchannels; outch += 3, koutch++) {
                    __m256d vq = _mm256_setzero_pd();

                    for (unsigned int kz = 0, iz = oz * stride; kz < kdepth; kz++, iz++) {
                        for (unsigned int ky = 0, iy = oy * stride; ky < kheight; ky++, iy++) {
                            for (unsigned int kx = 0, ix = ox * stride; kx < kwidth; kx++, ix++) {
                                for (unsigned int inch = 0, kinch = 0; inch < inchannels; inch += 3, kinch += 4) {
                                    __m256d v = _mm256_cvtps_pd(_mm_maskload_ps(inmap_ptr + inch + inchannels * (ix + inwidth * (iy + inheight * iz)), mask3));
                                    __m256d q = _mm256_cvtps_pd(_mm_load_ps(kernel_ptr + kinch + kernelinchannels * (koutch + kerneloutchannels * (kx + kwidth * (ky + kheight * kz)))));

                                    vq = _mm256_add_pd(_mm256_trivectormulgrad_pd(v, q), vq);
                                }
                            }
                        }
                    }

                    _mm_maskstore_ps(outmap_ptr + outch + outchannels * (ox + outwidth * (oy + outheight * oz)), mask3, _mm256_cvtpd_ps(vq));
                }
            }
        }
    }
}

void TensorShaderAvxBackend::Trivector::Convolution3D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth, unsigned int inheight, unsigned int indepth,
                                                      unsigned int batch, unsigned int th, unsigned int kwidth, unsigned int kheight, unsigned int kdepth, unsigned int stride, bool gradmode,
                                                      AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap) {

    Util::CheckDuplicateArray(inmap, kernel, outmap);

    if (inchannels % 3 != 0 || outchannels % 3 != 0) {
        throw gcnew System::ArgumentException();
    }

    if (th >= batch) {
        throw gcnew System::ArgumentException();
    }

    unsigned int outwidth = inwidth + 1 - kwidth;
    unsigned int outheight = inheight + 1 - kheight;
    unsigned int outdepth = indepth + 1 - kdepth;

    Util::CheckLength(inchannels * inwidth * inheight * indepth * batch, inmap);
    Util::CheckLength(outchannels * outwidth * outheight * outdepth * batch, outmap);
    Util::CheckLength(inchannels * outchannels * kwidth * kheight * kdepth * 4 / 9, kernel);

    float* inmap_ptr = (float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());
    float* kernel_ptr = (float*)(kernel->Ptr.ToPointer());

    if (gradmode) {
        trivector_convolution_3d_grad(inchannels, outchannels, 
                                      inwidth, outwidth, kwidth,
                                      inheight, outheight, kheight,
                                      indepth, outdepth, kdepth,
                                      stride, th, inmap_ptr, outmap_ptr, kernel_ptr);
    }
    else {
        trivector_convolution_3d(inchannels, outchannels, 
                                 inwidth, outwidth, kwidth,
                                 inheight, outheight, kheight,
                                 indepth, outdepth, kdepth,
                                 stride, th, inmap_ptr, outmap_ptr, kernel_ptr);
    }
}
