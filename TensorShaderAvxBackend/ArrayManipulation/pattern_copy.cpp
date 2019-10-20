#include "../TensorShaderAvxBackend.h"

using namespace System;

void pattern_copy_1_3(unsigned int src_stride, unsigned int dst_stride, unsigned int copy_length, unsigned int slides, float* src_ptr, float* dst_ptr) {
    const unsigned int j = copy_length & ~3, k = copy_length - j;
    const __m128i mask = TensorShaderAvxBackend::masktable_m128(k);

    while (slides > 0) {
        __m128 x = _mm_maskload_ps(src_ptr, mask);

        _mm_maskstore_ps(dst_ptr, mask, x);
        
        src_ptr += src_stride;
        dst_ptr += dst_stride;
        slides--;
    }
}

void pattern_copy_4(unsigned int src_stride, unsigned int dst_stride, unsigned int copy_length, unsigned int slides, float* src_ptr, float* dst_ptr) {
    while (slides > 0) {
        __m128 x = _mm_loadu_ps(src_ptr);

        _mm_storeu_ps(dst_ptr, x);

        src_ptr += src_stride;
        dst_ptr += dst_stride;
        slides--;
    }
}

void pattern_copy_5_7(unsigned int src_stride, unsigned int dst_stride, unsigned int copy_length, unsigned int slides, float* src_ptr, float* dst_ptr) {
    const unsigned int j = copy_length & ~7, k = copy_length - j;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

    while (slides > 0) {
        __m256 x = _mm256_maskload_ps(src_ptr, mask);

        _mm256_maskstore_ps(dst_ptr, mask, x);

        src_ptr += src_stride;
        dst_ptr += dst_stride;
        slides--;
    }
}

void pattern_copy_8(unsigned int src_stride, unsigned int dst_stride, unsigned int copy_length, unsigned int slides, float* src_ptr, float* dst_ptr) {
    while (slides > 0) {
        __m256 x = _mm256_loadu_ps(src_ptr);

        _mm256_storeu_ps(dst_ptr, x);

        src_ptr += src_stride;
        dst_ptr += dst_stride;
        slides--;
    }
}

void pattern_copy_9_(unsigned int src_stride, unsigned int dst_stride, unsigned int copy_length, unsigned int slides, float* src_ptr, float* dst_ptr) {
    const unsigned int j = copy_length & ~7, k = copy_length - j;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

    while(slides > 0) {
        for (unsigned int i = 0; i < j; i += 8) {
            __m256 x = _mm256_loadu_ps(src_ptr + i);

            _mm256_storeu_ps(dst_ptr + i, x);
        }

        if (k > 0) {
            __m256 x = _mm256_maskload_ps(src_ptr + j, mask);

            _mm256_maskstore_ps(dst_ptr + j, mask, x);
        }

        src_ptr += src_stride;
        dst_ptr += dst_stride;
        slides--;
    }
}

void TensorShaderAvxBackend::ArrayManipulation::PatternCopy(unsigned int src_stride, unsigned int src_index, unsigned int dst_stride, unsigned int dst_index, unsigned int copy_length, unsigned int slides, cli::array<float>^ src, cli::array<float>^ dst) {

    if (copy_length <= 0) {
        return;
    }

    Util::CheckOutOfRange(0, src_stride * slides, src);
    Util::CheckOutOfRange(0, dst_stride * slides, dst);
    
    if (src_index + copy_length > src_stride) {
        throw gcnew System::ArgumentException();
    }
    if (dst_index + copy_length > dst_stride) {
        throw gcnew System::ArgumentException();
    }

    pin_ptr<float> pinptr_src = &src[0];
    pin_ptr<float> pinptr_dst = &dst[0];

    float* src_ptr = pinptr_src;
    float* dst_ptr = pinptr_dst;

    if (copy_length <= 3) {
        pattern_copy_1_3(src_stride, dst_stride, copy_length, slides, src_ptr + src_index, dst_ptr + dst_index);
    }
    else if (copy_length <= 4) {
        pattern_copy_4(src_stride, dst_stride, copy_length, slides, src_ptr + src_index, dst_ptr + dst_index);
    }
    else if (copy_length <= 7) {
        pattern_copy_5_7(src_stride, dst_stride, copy_length, slides, src_ptr + src_index, dst_ptr + dst_index);
    }
    else if (copy_length <= 8) {
        pattern_copy_8(src_stride, dst_stride, copy_length, slides, src_ptr + src_index, dst_ptr + dst_index);
    }
    else {
        pattern_copy_9_(src_stride, dst_stride, copy_length, slides, src_ptr + src_index, dst_ptr + dst_index);
    }
}


void TensorShaderAvxBackend::ArrayManipulation::PatternCopy(unsigned int src_offset, unsigned int src_stride, unsigned int src_index, unsigned int dst_offset, unsigned int dst_stride, unsigned int dst_index, unsigned int copy_length, unsigned int slides, cli::array<float>^ src, cli::array<float>^ dst) {

    if (copy_length <= 0) {
        return;
    }

    Util::CheckOutOfRange(src_offset, src_stride * slides, src);
    Util::CheckOutOfRange(dst_offset, dst_stride * slides, dst);

    if (src_index + copy_length > src_stride) {
        throw gcnew System::ArgumentException();
    }
    if (dst_index + copy_length > dst_stride) {
        throw gcnew System::ArgumentException();
    }

    pin_ptr<float> pinptr_src = &src[0];
    pin_ptr<float> pinptr_dst = &dst[0];

    float* src_ptr = pinptr_src + src_offset;
    float* dst_ptr = pinptr_dst + dst_offset;

    if (copy_length <= 3) {
        pattern_copy_1_3(src_stride, dst_stride, copy_length, slides, src_ptr + src_index, dst_ptr + dst_index);
    }
    else if (copy_length <= 4) {
        pattern_copy_4(src_stride, dst_stride, copy_length, slides, src_ptr + src_index, dst_ptr + dst_index);
    }
    else if (copy_length <= 7) {
        pattern_copy_5_7(src_stride, dst_stride, copy_length, slides, src_ptr + src_index, dst_ptr + dst_index);
    }
    else if (copy_length <= 8) {
        pattern_copy_8(src_stride, dst_stride, copy_length, slides, src_ptr + src_index, dst_ptr + dst_index);
    }
    else {
        pattern_copy_9_(src_stride, dst_stride, copy_length, slides, src_ptr + src_index, dst_ptr + dst_index);
    }
}
