#include "../../TensorShaderAvxBackend.h"

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

void trivector_mulqgrad(unsigned int length, float* v_ptr, float* u_ptr, float* q_ptr, float* p_ptr) {
    const __m128i mask3 = TensorShaderAvxBackend::masktable_m128(3);

    for (unsigned int i = 0, j = 0; i < length; i += 3, j += 4) {
        __m256d v = _mm256_cvtps_pd(_mm_maskload_ps(v_ptr + i, mask3));
        __m256d u = _mm256_cvtps_pd(_mm_maskload_ps(u_ptr + i, mask3));
        __m256d q = _mm256_cvtps_pd(_mm_load_ps(q_ptr + j));

        __m128 p = _mm256_cvtpd_ps(_mm256_trivectormulqgrad_pd(v, u, q));

        _mm_store_ps(p_ptr + j, p);
    }
}

void TensorShaderAvxBackend::Trivector::MulQGrad(unsigned int vectorlength, AvxArray<float>^ src_vector_value, AvxArray<float>^ src_vector_grad, AvxArray<float>^ src_quaternion, AvxArray<float>^ dst_quaternion) {

    Util::CheckDuplicateArray(src_vector_value, src_vector_grad, src_quaternion, dst_quaternion);

    if (vectorlength % 3 != 0) {
        throw gcnew System::ArgumentException();
    }

    unsigned int quaternionlength = vectorlength / 3 * 4;

    Util::CheckLength(vectorlength, src_vector_value, src_vector_grad);
    Util::CheckLength(quaternionlength, src_quaternion, dst_quaternion);

    float* v_ptr = (float*)(src_vector_value->Ptr.ToPointer());
    float* u_ptr = (float*)(src_vector_grad->Ptr.ToPointer());
    float* q_ptr = (float*)(src_quaternion->Ptr.ToPointer());
    float* p_ptr = (float*)(dst_quaternion->Ptr.ToPointer());

    trivector_mulqgrad(vectorlength, v_ptr, u_ptr, q_ptr, p_ptr);
}