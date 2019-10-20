#include "../TensorShaderAvxBackend.h"
#include <array>
#include <vector>
#include <bitset>
#include <intrin.h>

using namespace System;

void TensorShaderAvxBackend::Util::CheckOutOfRange(unsigned int index, unsigned int length, ...cli::array<cli::array<float>^>^ arrays){
    if (length <= 0) {
        return;
    }
    
    if (index + length < index) {
        throw gcnew System::OverflowException();
    }
    
    for each (cli::array<float>^ array in arrays){
        if (index >= (unsigned int)array->Length || index + length > (unsigned int)array->Length) {
            throw gcnew System::IndexOutOfRangeException();
        }
    }
}

void TensorShaderAvxBackend::Util::CheckDuplicateArray(... cli::array<cli::array<float>^>^ arrays) {
    for (int i = 0; i < arrays->Length; i++) {
        for (int j = 0; j < i; j++) {
            if (ReferenceEquals(arrays[i], arrays[j])) {
                throw gcnew System::ArgumentException();
            }
        }
    }
}


bool is_supported_avx() {
    std::array<int, 4> cpui;
    std::vector<std::array<int, 4>> data;
    std::bitset<32> f_1_ECX;

    __cpuid(cpui.data(), 0);
    int nids = cpui[0];

    for (int i = 0; i <= nids; i++){
        __cpuidex(cpui.data(), i, 0);
        data.push_back(cpui);
    }

    if (nids >= 1){
        f_1_ECX = data[1][2];
    }

    return f_1_ECX[28];
}

bool TensorShaderAvxBackend::Util::IsSupportedAVX::get() {
    return is_supported_avx();
}

bool is_supported_avx2() {
    std::array<int, 4> cpui;
    std::vector<std::array<int, 4>> data;
    std::bitset<32> f_7_EBX;

    __cpuid(cpui.data(), 0);
    int nids = cpui[0];

    for (int i = 0; i <= nids; i++) {
        __cpuidex(cpui.data(), i, 0);
        data.push_back(cpui);
    }

    if (nids >= 7) {
        f_7_EBX = data[7][1];
    }

    return f_7_EBX[5];
}

bool TensorShaderAvxBackend::Util::IsSupportedAVX2::get() {
    return is_supported_avx2();
}

bool is_supported_avx512f() {
    std::array<int, 4> cpui;
    std::vector<std::array<int, 4>> data;
    std::bitset<32> f_7_EBX;

    __cpuid(cpui.data(), 0);
    int nids = cpui[0];

    for (int i = 0; i <= nids; i++) {
        __cpuidex(cpui.data(), i, 0);
        data.push_back(cpui);
    }

    if (nids >= 7) {
        f_7_EBX = data[7][1];
    }

    return f_7_EBX[16];
}

bool TensorShaderAvxBackend::Util::IsSupportedAVX512F::get() {
    return is_supported_avx512f();
}

bool is_supported_fma() {
    std::array<int, 4> cpui;
    std::vector<std::array<int, 4>> data;
    std::bitset<32> f_1_ECX;

    __cpuid(cpui.data(), 0);
    int nids = cpui[0];

    for (int i = 0; i <= nids; i++) {
        __cpuidex(cpui.data(), i, 0);
        data.push_back(cpui);
    }

    if (nids >= 1) {
        f_1_ECX = data[1][2];
    }

    return f_1_ECX[12];
}

bool TensorShaderAvxBackend::Util::IsSupportedFMA::get() {
    return is_supported_fma();
}