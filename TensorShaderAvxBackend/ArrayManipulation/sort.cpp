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

void TensorShaderAvxBackend::ArrayManipulation::Sort(unsigned int stride, unsigned int axislength, unsigned int slides, AvxArray<float>^ src, AvxArray<float>^ dst) {

    Util::CheckDuplicateArray(src, dst);
    Util::CheckLength(stride * axislength * slides, src, dst);

    AvxArray<float>::Copy(src, dst, (long long)(stride * axislength * slides));

    if (axislength <= 1) {
        return;
    }

    float* dst_ptr = (float*)(dst->Ptr.ToPointer());

    if (stride > 1) {
        sort(stride, axislength, slides, dst_ptr);
    }
    else {
        sort_stride1(axislength, slides, dst_ptr);
    }
}