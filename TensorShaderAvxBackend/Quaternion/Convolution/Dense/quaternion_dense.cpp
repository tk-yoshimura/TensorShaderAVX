#include "../../../TensorShaderAvxBackend.h"

using namespace System;

__forceinline __m256d _mm256_quaternionmul_pd(__m256d u, __m256d v) {
    const __m256d sign_mpmp = _mm256_castsi256_pd(_mm256_setr_epi64x(0x8000000000000000ull, 0, 0x8000000000000000ull, 0));
    const __m256d sign_mppm = _mm256_castsi256_pd(_mm256_setr_epi64x(0x8000000000000000ull, 0, 0, 0x8000000000000000ull));
    const __m256d sign_mmpp = _mm256_castsi256_pd(_mm256_setr_epi64x(0x8000000000000000ull, 0x8000000000000000ull, 0, 0));

    __m256d u_xxxx = _mm256_permute4x64_pd(u, _MM_PERM_AAAA);
    __m256d u_yyyy = _mm256_permute4x64_pd(u, _MM_PERM_BBBB);
    __m256d u_zzzz = _mm256_permute4x64_pd(u, _MM_PERM_CCCC);
    __m256d u_wwww = _mm256_permute4x64_pd(u, _MM_PERM_DDDD);

    __m256d v_xyzw = v;
    __m256d v_yxwz = _mm256_xor_pd(sign_mpmp, _mm256_permute4x64_pd(v_xyzw, _MM_PERM_CDAB));
    __m256d v_zwxy = _mm256_xor_pd(sign_mppm, _mm256_permute4x64_pd(v_xyzw, _MM_PERM_BADC));
    __m256d v_wzyx = _mm256_xor_pd(sign_mmpp, _mm256_permute4x64_pd(v_xyzw, _MM_PERM_ABCD));
    
    return _mm256_fmadd_pd(u_xxxx, v_xyzw, _mm256_fmadd_pd(u_yyyy, v_yxwz, _mm256_fmadd_pd(u_zzzz, v_zwxy, _mm256_mul_pd(u_wwww, v_wzyx))));
}

__forceinline __m256d _mm256_quaternionmulgrad_pd(__m256d u, __m256d v) {
    const __m256d sign_pmmp = _mm256_castsi256_pd(_mm256_setr_epi64x(0, 0x8000000000000000ull, 0x8000000000000000ull, 0));
    const __m256d sign_ppmm = _mm256_castsi256_pd(_mm256_setr_epi64x(0, 0, 0x8000000000000000ull, 0x8000000000000000ull));
    const __m256d sign_pmpm = _mm256_castsi256_pd(_mm256_setr_epi64x(0, 0x8000000000000000ull, 0, 0x8000000000000000ull));

    __m256d u_xxxx = _mm256_permute4x64_pd(u, _MM_PERM_AAAA);
    __m256d u_yyyy = _mm256_permute4x64_pd(u, _MM_PERM_BBBB);
    __m256d u_zzzz = _mm256_permute4x64_pd(u, _MM_PERM_CCCC);
    __m256d u_wwww = _mm256_permute4x64_pd(u, _MM_PERM_DDDD);

    __m256d v_xyzw = v;
    __m256d v_yxwz = _mm256_xor_pd(sign_pmmp, _mm256_permute4x64_pd(v_xyzw, _MM_PERM_CDAB));
    __m256d v_zwxy = _mm256_xor_pd(sign_ppmm, _mm256_permute4x64_pd(v_xyzw, _MM_PERM_BADC));
    __m256d v_wzyx = _mm256_xor_pd(sign_pmpm, _mm256_permute4x64_pd(v_xyzw, _MM_PERM_ABCD));

    return _mm256_fmadd_pd(u_xxxx, v_xyzw, _mm256_fmadd_pd(u_yyyy, v_yxwz, _mm256_fmadd_pd(u_zzzz, v_zwxy, _mm256_mul_pd(u_wwww, v_wzyx))));
}

void quaternion_dense(unsigned int inchannels, unsigned int outchannels, unsigned int th, const float* __restrict inmap_ptr, float* __restrict outmap_ptr, const float* __restrict kernel_ptr) {
    const unsigned int inmap_offset = inchannels * th, outmap_offset = outchannels * th;

    inmap_ptr += inmap_offset;
    outmap_ptr += outmap_offset;

    for (unsigned int outch = 0, koutch = 0; outch < outchannels; outch += 4, koutch++) {
        __m256d uv = _mm256_setzero_pd();

        for (unsigned int inch = 0; inch < inchannels; inch += 4) {
            __m256d u = _mm256_cvtps_pd(_mm_load_ps(inmap_ptr + inch));
            __m256d v = _mm256_cvtps_pd(_mm_load_ps(kernel_ptr + inch + inchannels * koutch));
            
            uv = _mm256_add_pd(_mm256_quaternionmul_pd(u, v), uv);
        }

        _mm_store_ps(outmap_ptr + outch, _mm256_cvtpd_ps(uv));
    }
}

void quaternion_dense_grad(unsigned int inchannels, unsigned int outchannels, unsigned int th, const float* __restrict inmap_ptr, float* __restrict outmap_ptr, const float* __restrict kernel_ptr) {
    const unsigned int inmap_offset = inchannels * th, outmap_offset = outchannels * th;

    inmap_ptr += inmap_offset;
    outmap_ptr += outmap_offset;

    for (unsigned int outch = 0, koutch = 0; outch < outchannels; outch += 4, koutch++) {
        __m256d vu = _mm256_setzero_pd();

        for (unsigned int inch = 0; inch < inchannels; inch += 4) {
            __m256d u = _mm256_cvtps_pd(_mm_load_ps(inmap_ptr + inch));
            __m256d v = _mm256_cvtps_pd(_mm_load_ps(kernel_ptr + inch + inchannels * koutch));

            vu = _mm256_add_pd(_mm256_quaternionmulgrad_pd(v, u), vu);
        }

        _mm_store_ps(outmap_ptr + outch, _mm256_cvtpd_ps(vu));
    }
}

void TensorShaderAvxBackend::Quaternion::Dense(unsigned int inchannels, unsigned int outchannels, unsigned int batch, unsigned int th, bool gradmode,
                                               AvxArray<float>^ inmap, AvxArray<float>^ kernel, AvxArray<float>^ outmap){

    Util::CheckDuplicateArray(inmap, kernel, outmap);

    if (inchannels % 4 != 0 || outchannels % 4 != 0) {
        throw gcnew System::ArgumentException();
    }

    if (th >= batch) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckLength(inchannels * batch, inmap);
    Util::CheckLength(outchannels * batch, outmap);
    Util::CheckLength(inchannels * outchannels / 4, kernel);

    float* inmap_ptr = (float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());
    float* kernel_ptr = (float*)(kernel->Ptr.ToPointer());

    if (gradmode) {
        quaternion_dense_grad(inchannels, outchannels, th, inmap_ptr, outmap_ptr, kernel_ptr);
    }
    else {
        quaternion_dense(inchannels, outchannels, th, inmap_ptr, outmap_ptr, kernel_ptr);
    }
}
