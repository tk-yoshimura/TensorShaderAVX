#include "../../../TensorShaderAvxBackend.h"

using namespace System;

__forceinline __m256d _mm256_quaternionmulkernelgrad_pd(__m256d u, __m256d v) {
    const __m256d sign_pmpm = _mm256_castsi256_pd(_mm256_setr_epi64x(0, 0x8000000000000000ull, 0, 0x8000000000000000ull));
    const __m256d sign_pmmp = _mm256_castsi256_pd(_mm256_setr_epi64x(0, 0x8000000000000000ull, 0x8000000000000000ull, 0));
    const __m256d sign_ppmm = _mm256_castsi256_pd(_mm256_setr_epi64x(0, 0, 0x8000000000000000ull, 0x8000000000000000ull));

    __m256d u_xxxx = _mm256_permute4x64_pd(u, _MM_PERM_AAAA);
    __m256d u_yyyy = _mm256_permute4x64_pd(u, _MM_PERM_BBBB);
    __m256d u_zzzz = _mm256_permute4x64_pd(u, _MM_PERM_CCCC);
    __m256d u_wwww = _mm256_permute4x64_pd(u, _MM_PERM_DDDD);

    __m256d v_xyzw = v;
    __m256d v_yxwz = _mm256_xor_pd(sign_pmpm, _mm256_permute4x64_pd(v_xyzw, _MM_PERM_CDAB));
    __m256d v_zwxy = _mm256_xor_pd(sign_pmmp, _mm256_permute4x64_pd(v_xyzw, _MM_PERM_BADC));
    __m256d v_wzyx = _mm256_xor_pd(sign_ppmm, _mm256_permute4x64_pd(v_xyzw, _MM_PERM_ABCD));

    return _mm256_fmadd_pd(u_xxxx, v_xyzw, _mm256_fmadd_pd(u_yyyy, v_yxwz, _mm256_fmadd_pd(u_zzzz, v_zwxy, _mm256_mul_pd(u_wwww, v_wzyx))));
}

void quaternion_kernelproduct_1d(unsigned int inchannels, unsigned int outchannels, 
                                 unsigned int inwidth, unsigned int outwidth, unsigned int kwidth,
                                 unsigned int stride, unsigned int batch, unsigned int outch, 
                                 const float* __restrict inmap_ptr, float* __restrict outmap_ptr, const float* __restrict kernel_ptr) {

    const unsigned int kerneloutchannels = outchannels / 4, koutch = outch / 4;
    
    for (unsigned int inch = 0; inch < inchannels; inch += 4) {
        for (unsigned int kx = 0; kx < kwidth; kx++) {
            __m256d uv = _mm256_setzero_pd();

            for (unsigned int th = 0; th < batch; th++) {
                for (unsigned int ox = 0, ix = kx; ox < outwidth; ox++, ix += stride) {
                    __m256d u = _mm256_cvtps_pd(_mm_load_ps(inmap_ptr + inch + inchannels * (ix + inwidth * th)));
                    __m256d v = _mm256_cvtps_pd(_mm_load_ps(outmap_ptr + outch + outchannels * (ox + outwidth * th)));

                    uv = _mm256_add_pd(_mm256_quaternionmulkernelgrad_pd(u, v), uv);
                }
            }

            _mm_store_ps(kernel_ptr + inch + inchannels * (koutch + kerneloutchannels * kx), _mm256_cvtpd_ps(uv));
        }
    }
}

void quaternion_kernelproduct_1d_transpose(unsigned int inchannels, unsigned int outchannels, 
                                           unsigned int inwidth, unsigned int outwidth, unsigned int kwidth, 
                                           unsigned int stride, unsigned int batch, unsigned int outch, 
                                           const float* __restrict inmap_ptr, float* __restrict outmap_ptr, const float* __restrict kernel_ptr) {

    const unsigned int kerneloutchannels = outchannels / 4, koutch = outch / 4;

    for (unsigned int inch = 0; inch < inchannels; inch += 4) {
        for (unsigned int kx = 0; kx < kwidth; kx++) {
            __m256d vu = _mm256_setzero_pd();

            for (unsigned int th = 0; th < batch; th++) {
                for (unsigned int ox = 0, ix = kx; ox < outwidth; ox++, ix += stride) {
                    __m256d u = _mm256_cvtps_pd(_mm_load_ps(inmap_ptr + inch + inchannels * (ix + inwidth * th)));
                    __m256d v = _mm256_cvtps_pd(_mm_load_ps(outmap_ptr + outch + outchannels * (ox + outwidth * th)));

                    vu = _mm256_add_pd(_mm256_quaternionmulkernelgrad_pd(v, u), vu);
                }
            }

            _mm_store_ps(kernel_ptr + inch + inchannels * (koutch + kerneloutchannels * kx), _mm256_cvtpd_ps(vu));
        }
    }
}


void TensorShaderAvxBackend::Quaternion::KernelProduct1D(unsigned int inchannels, unsigned int outchannels, unsigned int inwidth, 
                                                         unsigned int batch, unsigned int outch, unsigned int kwidth, unsigned int stride, bool transpose,
                                                         AvxArray<float>^ inmap, AvxArray<float>^ outmap, AvxArray<float>^ kernel) {

    Util::CheckDuplicateArray(inmap, kernel, outmap);
    
    if (inchannels % 4 != 0 || outchannels % 4 != 0 || outch % 4 != 0) {
        throw gcnew System::ArgumentException();
    }

    if (outch >= outchannels) {
        throw gcnew System::ArgumentException();
    }

    unsigned int outwidth = inwidth + 1 - kwidth;

    Util::CheckLength(inchannels * inwidth * batch, inmap);
    Util::CheckLength(outchannels * outwidth * batch, outmap);
    Util::CheckLength(inchannels * outchannels * kwidth / 4, kernel);

    float* inmap_ptr = (float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());
    float* kernel_ptr = (float*)(kernel->Ptr.ToPointer());

    if (transpose) {
        quaternion_kernelproduct_1d_transpose(inchannels, outchannels, 
                                              inwidth, outwidth, kwidth,
                                              stride, batch, outch, 
                                              inmap_ptr, outmap_ptr, kernel_ptr);
    }
    else {
        quaternion_kernelproduct_1d(inchannels, outchannels, 
                                    inwidth, outwidth, kwidth,
                                    stride, batch, outch, 
                                    inmap_ptr, outmap_ptr, kernel_ptr);
    }
}
