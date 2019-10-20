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

void quaternion_kernelproduct_dense(unsigned int inchannels, unsigned int outchannels, unsigned int batch, unsigned int outch, float* inmap_ptr, float* outmap_ptr, float* kernel_ptr) {
    const unsigned int koutch = outch / 4;
    
    for (unsigned int inch = 0; inch < inchannels; inch += 4) {
        __m256d uv = _mm256_setzero_pd();
            
        for (unsigned int th = 0; th < batch; th++) {
            __m256d u = _mm256_cvtps_pd(_mm_loadu_ps(inmap_ptr + inch + inchannels * th));
            __m256d v = _mm256_cvtps_pd(_mm_loadu_ps(outmap_ptr + outch + outchannels * th));

            uv = _mm256_add_pd(_mm256_quaternionmulkernelgrad_pd(u, v), uv);
        }

        _mm_storeu_ps(kernel_ptr + inch + inchannels * koutch, _mm256_cvtpd_ps(uv));
    }
}

void quaternion_kernelproduct_dense_transpose(unsigned int inchannels, unsigned int outchannels, unsigned int batch, unsigned int outch, float* inmap_ptr, float* outmap_ptr, float* kernel_ptr) {
    const unsigned int koutch = outch / 4;
    
    for (unsigned int inch = 0; inch < inchannels; inch += 4) {
        __m256d vu = _mm256_setzero_pd();

        for (unsigned int th = 0; th < batch; th++) {
            __m256d u = _mm256_cvtps_pd(_mm_loadu_ps(inmap_ptr + inch + inchannels * th));
            __m256d v = _mm256_cvtps_pd(_mm_loadu_ps(outmap_ptr + outch + outchannels * th));

            vu = _mm256_add_pd(_mm256_quaternionmulkernelgrad_pd(v, u), vu);
        }

        _mm_storeu_ps(kernel_ptr + inch + inchannels * koutch, _mm256_cvtpd_ps(vu));
    }
}


void TensorShaderAvxBackend::Quaternion::KernelProductDense(unsigned int inchannels, unsigned int outchannels, unsigned int batch, unsigned int outch, bool transpose,
                                                            cli::array<float>^ inmap, cli::array<float>^ outmap, cli::array<float>^ kernel) {

    Util::CheckDuplicateArray(inmap, kernel, outmap);

    if (inchannels % 4 != 0 || outchannels % 4 != 0 || outch % 4 != 0) {
        throw gcnew System::ArgumentException();
    }

    if (inchannels * batch > (unsigned int)inmap->Length) {
        throw gcnew System::ArgumentException();
    }
    if (outchannels * batch > (unsigned int)outmap->Length) {
        throw gcnew System::ArgumentException();
    }
    if (inchannels * outchannels / 4 > (unsigned int)kernel->Length) {
        throw gcnew System::ArgumentException();
    }
    if (outch >= outchannels) {
        throw gcnew System::ArgumentException();
    }

    pin_ptr<float> pinptr_inmap = &inmap[0];
    pin_ptr<float> pinptr_outmap = &outmap[0];
    pin_ptr<float> pinptr_kernel = &kernel[0];

    float* inmap_ptr = pinptr_inmap;
    float* outmap_ptr = pinptr_outmap;
    float* kernel_ptr = pinptr_kernel;

    if (transpose) {
        quaternion_kernelproduct_dense_transpose(inchannels, outchannels, batch, outch, inmap_ptr, outmap_ptr, kernel_ptr);
    }
    else {
        quaternion_kernelproduct_dense(inchannels, outchannels, batch, outch, inmap_ptr, outmap_ptr, kernel_ptr);
    }
}
