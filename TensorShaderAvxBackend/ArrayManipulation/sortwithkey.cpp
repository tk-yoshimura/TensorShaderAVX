#include "../TensorShaderAvxBackend.h"

using namespace System;

void sortwithkey(unsigned int stride, unsigned int axislength, unsigned int slides, float* key_ptr, float* val_ptr) {
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

                    if (key_ptr[a] > key_ptr[b]) {
                        temp = key_ptr[a]; key_ptr[a] = key_ptr[b]; key_ptr[b] = temp;
                        temp = val_ptr[a]; val_ptr[a] = val_ptr[b]; val_ptr[b] = temp;
                        is_swapped = true;
                    }
                }
            }
        }
    }
}

void sortwithkey_stride1(unsigned int axislength, unsigned int slides, float* key_ptr, float* val_ptr) {
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

                if (key_ptr[a] > key_ptr[b]) {
                    temp = key_ptr[a]; key_ptr[a] = key_ptr[b]; key_ptr[b] = temp;
                    temp = val_ptr[a]; val_ptr[a] = val_ptr[b]; val_ptr[b] = temp;
                    is_swapped = true;
                }
            }
        }
    }
}

void TensorShaderAvxBackend::ArrayManipulation::SortWithKey(unsigned int stride, unsigned int axislength, unsigned int slides, AvxArray<float>^ src_key, AvxArray<float>^ src_value, AvxArray<float>^ dst_key, AvxArray<float>^ dst_value) {

    Util::CheckDuplicateArray(src_key, src_value, dst_key, dst_value);
    Util::CheckLength(stride * axislength * slides, src_key, src_value, dst_key, dst_value);

    AvxArray<float>::Copy(src_key, dst_key, (long long)(stride * axislength * slides));
    AvxArray<float>::Copy(src_value, dst_value, (long long)(stride * axislength * slides));

    if (axislength <= 1) {
        return;
    }

    float* key_ptr = (float*)(dst_key->Ptr.ToPointer());
    float* val_ptr = (float*)(dst_value->Ptr.ToPointer());

    if (stride > 1) {
        sortwithkey(stride, axislength, slides, key_ptr, val_ptr);
    }
    else {
        sortwithkey_stride1(axislength, slides, key_ptr, val_ptr);
    }
}