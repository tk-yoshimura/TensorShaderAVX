#include "../TensorShaderAvxBackend.h"
#include <vector>

using namespace System;

__forceinline __m256 _mm256_onehotvector_ps(__m256 x, __m256 idx) {
    union {
        float f;
        unsigned int i;
    }m32;
    m32.i = 0x7FFFFFFFu;

    const __m256 bitmask = _mm256_set1_ps(m32.f);

    __m256 y = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(_mm256_set1_ps(1), _mm256_and_ps(bitmask, _mm256_sub_ps(x, idx))));

    return y;
}

void onehotvector(unsigned int length, unsigned int channels, const float* __restrict src_ptr, float* __restrict dst_ptr) {
    const unsigned int j = channels & ~7u, k = channels - j;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

    std::vector<float> ch = std::vector<float>(channels);
    for (unsigned int i = 0; i < channels; i++) {
        ch[i] = (float)i;
    }
    
    for (unsigned int n = 0; n < length; n++) {
        __m256 idx = _mm256_set1_ps(src_ptr[n]);

        for (unsigned int i = 0; i < j; i += 8) {
            __m256 x = _mm256_loadu_ps(ch.data() + i);

            __m256 y = _mm256_onehotvector_ps(x, idx);

            _mm256_storeu_ps(dst_ptr + i, y);
        }

        if (k > 0) {
            __m256 x = _mm256_maskload_ps(ch.data() + j, mask);

            __m256 y = _mm256_onehotvector_ps(x, idx);

            _mm256_maskstore_ps(dst_ptr + j, mask, y);
        }

        dst_ptr += channels;
    }
}

void TensorShaderAvxBackend::Indexer::OneHotVector(unsigned int length, unsigned int channels, AvxArray<float>^ src, AvxArray<float>^ dst) {

    Util::CheckDuplicateArray(src, dst);

    Util::CheckLength(length, src);
    Util::CheckLength(length * channels, dst);
    
    float* src_ptr = (float*)(src->Ptr.ToPointer());
    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    onehotvector(length, channels, src_ptr, dst_ptr);
}