#include "../TensorShaderAvxBackend.h"

using namespace System;

void broadcast_1(unsigned int src_length, float* src_ptr, unsigned int dst_length, float* dst_ptr, unsigned int slides) {
    const unsigned int j = dst_length & ~7u, k = dst_length - j;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

    while (slides > 0) {
        __m256 x = _mm256_set1_ps(src_ptr[0]);

        for (unsigned int i = 0; i < j; i += 8) {
            _mm256_storeu_ps(dst_ptr + i, x);
        }

        if (k > 0) {
            _mm256_maskstore_ps(dst_ptr + j, mask, x);
        }

        src_ptr += src_length;
        dst_ptr += dst_length;
        slides--;
    }
}

void broadcast_2(unsigned int src_length, float* src_ptr, unsigned int dst_length, float* dst_ptr, unsigned int slides) {
    const unsigned int j = dst_length & ~7u, k = dst_length - j;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

    while (slides > 0) {
        __m256 x = _mm256_setr_ps(src_ptr[0], src_ptr[1], src_ptr[0], src_ptr[1], src_ptr[0], src_ptr[1], src_ptr[0], src_ptr[1]);

        for (unsigned int i = 0; i < j; i += 8) {
            _mm256_storeu_ps(dst_ptr + i, x);
        }

        if (k > 0) {
            _mm256_maskstore_ps(dst_ptr + j, mask, x);
        }

        src_ptr += src_length;
        dst_ptr += dst_length;
        slides--;
    }
}

void broadcast_3(unsigned int src_length, float* src_ptr, unsigned int dst_length, float* dst_ptr, unsigned int slides) {
    const unsigned int j = dst_length / 6 * 6, k = dst_length - j;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);
    const __m256i mask6 = TensorShaderAvxBackend::masktable_m256(6);

    while (slides > 0) {
        __m256 x = _mm256_setr_ps(src_ptr[0], src_ptr[1], src_ptr[2], src_ptr[0], src_ptr[1], src_ptr[2], src_ptr[0], src_ptr[1]);

        for (unsigned int i = 0; i < j; i += 6) {
            _mm256_maskstore_ps(dst_ptr + i, mask6, x);
        }

        if (k > 0) {
            _mm256_maskstore_ps(dst_ptr + j, mask, x);
        }

        src_ptr += src_length;
        dst_ptr += dst_length;
        slides--;
    }
}

void broadcast_4(unsigned int src_length, float* src_ptr, unsigned int dst_length, float* dst_ptr, unsigned int slides) {
    const unsigned int j = dst_length & ~7u, k = dst_length - j;
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(k);

    while (slides > 0) {
        __m256 x = _mm256_setr_ps(src_ptr[0], src_ptr[1], src_ptr[2], src_ptr[3], src_ptr[0], src_ptr[1], src_ptr[2], src_ptr[3]);

        for (unsigned int i = 0; i < j; i += 8) {
            _mm256_storeu_ps(dst_ptr + i, x);
        }

        if (k > 0) {
            _mm256_maskstore_ps(dst_ptr + j, mask, x);
        }

        src_ptr += src_length;
        dst_ptr += dst_length;
        slides--;
    }
}

void broadcast_5_7(unsigned int src_length, float* src_ptr, unsigned int dst_length, float* dst_ptr, unsigned int slides) {
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(src_length);

    while (slides > 0) {
        __m256 x = _mm256_maskload_ps(src_ptr, mask);

        for (unsigned int i = 0; i < dst_length; i += src_length) {
            _mm256_maskstore_ps(dst_ptr + i, mask, x);
        }

        src_ptr += src_length;
        dst_ptr += dst_length;
        slides--;
    }
}

void broadcast_8(unsigned int src_length, float* src_ptr, unsigned int dst_length, float* dst_ptr, unsigned int slides) {
    while (slides > 0) {
        __m256 x = _mm256_loadu_ps(src_ptr);

        for (unsigned int i = 0; i < dst_length; i += src_length) {
            _mm256_storeu_ps(dst_ptr + i, x);
        }

        src_ptr += src_length;
        dst_ptr += dst_length;
        slides--;
    }
}

void broadcast_9_(unsigned int src_length, float* src_ptr, unsigned int dst_length, float* dst_ptr, unsigned int slides) {
    const unsigned int r = src_length & ~7u, s = src_length - r;    
    const __m256i mask = TensorShaderAvxBackend::masktable_m256(s);

    while (slides > 0) {
        for (unsigned int i = 0; i < dst_length; i += src_length) {
            for (unsigned int j = 0; j < r; j += 8) {
                __m256 x = _mm256_loadu_ps(src_ptr + j);

                _mm256_storeu_ps(dst_ptr + i + j, x);
            }
            if (s > 0) {
                __m256 x = _mm256_maskload_ps(src_ptr + r, mask);

                _mm256_maskstore_ps(dst_ptr + i + r, mask, x);
            }
        }

        src_ptr += src_length;
        dst_ptr += dst_length;
        slides--;
    }
}

void TensorShaderAvxBackend::ArrayManipulation::Broadcast(unsigned int src_index, unsigned int src_length, cli::array<float>^ src, unsigned int dst_index, unsigned int dst_length, cli::array<float>^ dst, unsigned int slides) {

    Util::CheckDuplicateArray(src, dst);

    if (src_length == 0 && dst_length == 0) {
        return;
    }

    if (slides == 0) {
        return;
    }

    if (((src_length * slides) / slides) != src_length) {
        throw gcnew System::OverflowException();
    }

    if (((dst_length * slides) / slides) != dst_length) {
        throw gcnew System::OverflowException();
    }

    Util::CheckOutOfRange(src_index, src_length * slides, src);
    Util::CheckOutOfRange(dst_index, dst_length * slides, dst);

    if (src_length < 1 || dst_length < 1 || dst_length % src_length != 0) {
        throw gcnew System::ArgumentException();
    }

    pin_ptr<float> pinptr_src = &src[0];
    pin_ptr<float> pinptr_dst = &dst[0];

    float* src_ptr = pinptr_src;
    float* dst_ptr = pinptr_dst;

    if (src_length <= 1) {
        broadcast_1(src_length, src_ptr + src_index, dst_length, dst_ptr + dst_index, slides);
    }
    else if (src_length <= 2) {
        broadcast_2(src_length, src_ptr + src_index, dst_length, dst_ptr + dst_index, slides);
    }
    else if (src_length <= 3) {
        broadcast_3(src_length, src_ptr + src_index, dst_length, dst_ptr + dst_index, slides);
    }
    else if (src_length <= 4) {
        broadcast_4(src_length, src_ptr + src_index, dst_length, dst_ptr + dst_index, slides);
    }
    else if (src_length <= 7) {
        broadcast_5_7(src_length, src_ptr + src_index, dst_length, dst_ptr + dst_index, slides);
    }
    else if (src_length <= 8) {
        broadcast_8(src_length, src_ptr + src_index, dst_length, dst_ptr + dst_index, slides);
    }
    else {
        broadcast_9_(src_length, src_ptr + src_index, dst_length, dst_ptr + dst_index, slides);
    }
}