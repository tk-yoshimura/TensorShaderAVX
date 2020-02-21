#include "../../../TensorShaderAvxBackend.h"

using namespace System;

__forceinline __m256d _mm256_trivectormulqgrad_pd(__m256d v, __m256d u, __m256d q) {
    const __m256d sign_ppmm = _mm256_castsi256_pd(_mm256_setr_epi64x(0, 0, 0x8000000000000000ull, 0x8000000000000000ull));
    const __m256d sign_pmpm = _mm256_castsi256_pd(_mm256_setr_epi64x(0, 0x8000000000000000ull, 0, 0x8000000000000000ull));
    const __m256d sign_pmmp = _mm256_castsi256_pd(_mm256_setr_epi64x(0, 0x8000000000000000ull, 0x8000000000000000ull, 0));

    __m256d vxq_xyzw = _mm256_mul_pd(_mm256_permute4x64_pd(v, _MM_PERM_AAAA), q);
    __m256d vyq_xyzw = _mm256_mul_pd(_mm256_permute4x64_pd(v, _MM_PERM_BBBB), q);
    __m256d vzq_xyzw = _mm256_mul_pd(_mm256_permute4x64_pd(v, _MM_PERM_CCCC), q);

    __m256d vzq_zwxy = _mm256_permute4x64_pd(vzq_xyzw, _MM_PERM_BADC);
    __m256d vyq_wzyx = _mm256_permute4x64_pd(vyq_xyzw, _MM_PERM_ABCD);
    __m256d vx = _mm256_add_pd(vzq_zwxy, _mm256_sub_pd(_mm256_xor_pd(sign_ppmm, vxq_xyzw), _mm256_xor_pd(sign_pmmp, vyq_wzyx)));
    __m256d uvx = _mm256_mul_pd(_mm256_permute4x64_pd(u, _MM_PERM_AAAA), vx);

    __m256d vxq_wzyx = _mm256_permute4x64_pd(vxq_xyzw, _MM_PERM_ABCD);
    __m256d vzq_yxwz = _mm256_permute4x64_pd(vzq_xyzw, _MM_PERM_CDAB);
    __m256d vy = _mm256_add_pd(vxq_wzyx, _mm256_sub_pd(_mm256_xor_pd(sign_pmpm, vyq_xyzw), _mm256_xor_pd(sign_ppmm, vzq_yxwz)));
    __m256d uvy = _mm256_mul_pd(_mm256_permute4x64_pd(u, _MM_PERM_BBBB), vy);

    __m256d vyq_yxwz = _mm256_permute4x64_pd(vyq_xyzw, _MM_PERM_CDAB);
    __m256d vxq_zwxy = _mm256_permute4x64_pd(vxq_xyzw, _MM_PERM_BADC);
    __m256d vz = _mm256_add_pd(vyq_yxwz, _mm256_sub_pd(_mm256_xor_pd(sign_pmmp, vzq_xyzw), _mm256_xor_pd(sign_pmpm, vxq_zwxy)));
    __m256d uvz = _mm256_mul_pd(_mm256_permute4x64_pd(u, _MM_PERM_CCCC), vz);

    __m256d p_half = _mm256_add_pd(uvx, _mm256_add_pd(uvy, uvz));

    __m256d p = _mm256_add_pd(p_half, p_half);

    return p;
}

void trivector_kernelproduct_dense(unsigned int inchannels, unsigned int outchannels, unsigned int batch, unsigned int outch, const float* __restrict inmap_ptr, float* __restrict outmap_ptr, float* kernelvalue_ptr, float* kernelgrad_ptr) {
    const unsigned int kernelinchannels = inchannels / 3 * 4, koutch = outch / 3;
    const __m128i mask3 = TensorShaderAvxBackend::masktable_m128(3);
    
    for (unsigned int inch = 0, kinch = 0; inch < inchannels; inch += 3, kinch += 4) {
        __m256d uvq = _mm256_setzero_pd();
        __m256d q = _mm256_cvtps_pd(_mm_loadu_ps(kernelvalue_ptr + kinch + kernelinchannels * koutch));
            
        for (unsigned int th = 0; th < batch; th++) {
            __m256d u = _mm256_cvtps_pd(_mm_maskload_ps(inmap_ptr + inch + inchannels * th, mask3));
            __m256d v = _mm256_cvtps_pd(_mm_maskload_ps(outmap_ptr + outch + outchannels * th, mask3));

            uvq = _mm256_add_pd(_mm256_trivectormulqgrad_pd(u, v, q), uvq);
        }

        _mm_storeu_ps(kernelgrad_ptr + kinch + kernelinchannels * koutch, _mm256_cvtpd_ps(uvq));
    }
}

void trivector_kernelproduct_dense_transpose(unsigned int inchannels, unsigned int outchannels, unsigned int batch, unsigned int outch, const float* __restrict inmap_ptr, float* __restrict outmap_ptr, float* kernelvalue_ptr, float* kernelgrad_ptr) {
    const unsigned int kernelinchannels = inchannels / 3 * 4, koutch = outch / 3;
    const __m128i mask3 = TensorShaderAvxBackend::masktable_m128(3);

    for (unsigned int inch = 0, kinch = 0; inch < inchannels; inch += 3, kinch += 4) {
        __m256d vuq = _mm256_setzero_pd();
        __m256d q = _mm256_cvtps_pd(_mm_loadu_ps(kernelvalue_ptr + kinch + kernelinchannels * koutch));

        for (unsigned int th = 0; th < batch; th++) {
            __m256d u = _mm256_cvtps_pd(_mm_maskload_ps(inmap_ptr + inch + inchannels * th, mask3));
            __m256d v = _mm256_cvtps_pd(_mm_maskload_ps(outmap_ptr + outch + outchannels * th, mask3));

            vuq = _mm256_add_pd(_mm256_trivectormulqgrad_pd(v, u, q), vuq);
        }

        _mm_storeu_ps(kernelgrad_ptr + kinch + kernelinchannels * koutch, _mm256_cvtpd_ps(vuq));
    }
}


void TensorShaderAvxBackend::Trivector::KernelProductDense(unsigned int inchannels, unsigned int outchannels, unsigned int batch, unsigned int outch, bool transpose,
                                                           AvxArray<float>^ inmap, AvxArray<float>^ outmap, AvxArray<float>^ kernel_value, AvxArray<float>^ kernel_grad) {

    Util::CheckDuplicateArray(inmap, outmap, kernel_value, kernel_grad);

    if (inchannels % 3 != 0 || outchannels % 3 != 0 || outch % 3 != 0) {
        throw gcnew System::ArgumentException();
    }

    if (outch >= outchannels) {
        throw gcnew System::ArgumentException();
    }

    Util::CheckLength(inchannels * batch, inmap);
    Util::CheckLength(outchannels * batch, outmap);
    Util::CheckLength(inchannels * outchannels * 4 / 9, kernel_value, kernel_grad);

    const float* inmap_ptr = (const float*)(inmap->Ptr.ToPointer());
    float* outmap_ptr = (float*)(outmap->Ptr.ToPointer());
    float* kernelvalue_ptr = (float*)(kernel_value->Ptr.ToPointer());
    float* kernelgrad_ptr = (float*)(kernel_grad->Ptr.ToPointer());

    if (transpose) {
        trivector_kernelproduct_dense_transpose(inchannels, outchannels, batch, outch, inmap_ptr, outmap_ptr, kernelvalue_ptr, kernelgrad_ptr);
    }
    else {
        trivector_kernelproduct_dense(inchannels, outchannels, batch, outch, inmap_ptr, outmap_ptr, kernelvalue_ptr, kernelgrad_ptr);
    }
}
