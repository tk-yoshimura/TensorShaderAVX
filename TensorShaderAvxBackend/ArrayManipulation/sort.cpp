#include "../TensorShaderAvxBackend.h"

using namespace System;

void sort(unsigned int stride, unsigned int axislength, unsigned int slides, float* ref_ptr) {
    float temp;

    for (unsigned int k = 0; k < slides; k++) {
        for (unsigned int i = 0; i < stride; i++) {
            int idx = i + k * stride * axislength;

            unsigned int h = axislength;
            bool is_swapped = false;

            while (h > 1 || is_swapped) {
                if (h > 1) {
                    h = h * 10 / 13;
                }
                if (h == 9 || h == 10) {
                    h = 11;
                }

                is_swapped = false;
                for (unsigned int s = 0; s < axislength - h; s++) {
                    unsigned int a = idx + s * stride, b = a + h * stride;

                    if (ref_ptr[a] > ref_ptr[b]) {
                        temp = ref_ptr[a]; ref_ptr[a] = ref_ptr[b]; ref_ptr[b] = temp;
                        is_swapped = true;
                    }
                }
            }
        }
    }
}

void sort_stride1(unsigned int axislength, unsigned int slides, float* ref_ptr) {
    float temp;

    for (unsigned int k = 0, idx = 0; k < slides; k++, idx += axislength) {

        unsigned int h = axislength;
        bool is_swapped = false;

        while (h > 1 || is_swapped) {
            if (h > 1) {
                h = h * 10 / 13;
            }
            if (h == 9 || h == 10) {
                h = 11;
            }

            is_swapped = false;
            for (unsigned int s = 0; s < axislength - h; s++) {
                unsigned int a = idx + s, b = a + h;

                if (ref_ptr[a] > ref_ptr[b]) {
                    temp = ref_ptr[a]; ref_ptr[a] = ref_ptr[b]; ref_ptr[b] = temp;
                    is_swapped = true;
                }
            }
        }
    }
}

void TensorShaderAvxBackend::ArrayManipulation::Sort(unsigned int stride, unsigned int axislength, unsigned int slides, cli::array<float>^ src, cli::array<float>^ dst) {

    Util::CheckDuplicateArray(src, dst);
    Util::CheckOutOfRange(0, stride * axislength * slides, src, dst);

    Array::Copy(src, dst, (long long)(stride * axislength * slides));

    if (axislength <= 1) {
        return;
    }

    pin_ptr<float> pinptr_dst = &dst[0];

    float* dst_ptr = pinptr_dst;

    if (stride > 1) {
        sort(stride, axislength, slides, dst_ptr);
    }
    else {
        sort_stride1(axislength, slides, dst_ptr);
    }
}