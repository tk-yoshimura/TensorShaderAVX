#include "../../TensorShaderAvxBackend.h"

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

void trivector_mul(unsigned int length, float* v_ptr, float* q_ptr, float* u_ptr) {
    const __m128i mask3 = TensorShaderAvxBackend::masktable_m128(3);

    for (unsigned int i = 0, j = 0; i < length; i += 3, j += 4) {
        __m256d v = _mm256_cvtps_pd(_mm_maskload_ps(v_ptr + i, mask3));
        __m256d q = _mm256_cvtps_pd(_mm_load_ps(q_ptr + j));

        __m128 u = _mm256_cvtpd_ps(_mm256_trivectormul_pd(v, q));

        _mm_maskstore_ps(u_ptr + i, mask3, u);
    }
}

void TensorShaderAvxBackend::Trivector::Mul(unsigned int vectorlength, AvxArray<float>^ src_vector, AvxArray<float>^ src_quaternion, AvxArray<float>^ dst_vector) {
    
    Util::CheckDuplicateArray(src_vector, src_quaternion, dst_vector);

    if (vectorlength % 3 != 0) {
        throw gcnew System::ArgumentException();
    }

    unsigned int quaternionlength = vectorlength / 3 * 4;

    Util::CheckLength(vectorlength, src_vector, dst_vector);
    Util::CheckLength(quaternionlength, src_quaternion);
    
    float* v_ptr = (float*)(src_vector->Ptr.ToPointer());
    float* q_ptr = (float*)(src_quaternion->Ptr.ToPointer());
    float* u_ptr = (float*)(dst_vector->Ptr.ToPointer());

    trivector_mul(vectorlength, v_ptr, q_ptr, u_ptr);
}