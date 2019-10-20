#include "../TensorShaderAvxBackend.h"

TensorShaderAvxBackend::Elementwise::Elementwise() {
    if (!TensorShaderAvxBackend::Util::IsSupportedAVX || !TensorShaderAvxBackend::Util::IsSupportedAVX2) {
        throw gcnew System::PlatformNotSupportedException("AVX2 not supported on this platform.");
    }
}

TensorShaderAvxBackend::Channelwise::Channelwise() {
    if (!TensorShaderAvxBackend::Util::IsSupportedAVX || !TensorShaderAvxBackend::Util::IsSupportedAVX2) {
        throw gcnew System::PlatformNotSupportedException("AVX2 not supported on this platform.");
    }
}

TensorShaderAvxBackend::Batchwise::Batchwise() {
    if (!TensorShaderAvxBackend::Util::IsSupportedAVX || !TensorShaderAvxBackend::Util::IsSupportedAVX2) {
        throw gcnew System::PlatformNotSupportedException("AVX2 not supported on this platform.");
    }
}

TensorShaderAvxBackend::ArrayManipulation::ArrayManipulation() {
    if (!TensorShaderAvxBackend::Util::IsSupportedAVX || !TensorShaderAvxBackend::Util::IsSupportedAVX2) {
        throw gcnew System::PlatformNotSupportedException("AVX2 not supported on this platform.");
    }
}

TensorShaderAvxBackend::Indexer::Indexer() {
    if (!TensorShaderAvxBackend::Util::IsSupportedAVX || !TensorShaderAvxBackend::Util::IsSupportedAVX2) {
        throw gcnew System::PlatformNotSupportedException("AVX2 not supported on this platform.");
    }
}

TensorShaderAvxBackend::Aggregation::Aggregation() {
    if (!TensorShaderAvxBackend::Util::IsSupportedAVX || !TensorShaderAvxBackend::Util::IsSupportedAVX2) {
        throw gcnew System::PlatformNotSupportedException("AVX2 not supported on this platform.");
    }
}

TensorShaderAvxBackend::Pool::Pool() {
    if (!TensorShaderAvxBackend::Util::IsSupportedAVX || !TensorShaderAvxBackend::Util::IsSupportedAVX2) {
        throw gcnew System::PlatformNotSupportedException("AVX2 not supported on this platform.");
    }
}

TensorShaderAvxBackend::Transform::Transform() {
    if (!TensorShaderAvxBackend::Util::IsSupportedAVX || !TensorShaderAvxBackend::Util::IsSupportedAVX2) {
        throw gcnew System::PlatformNotSupportedException("AVX2 not supported on this platform.");
    }
}

TensorShaderAvxBackend::Padding::Padding() {
    if (!TensorShaderAvxBackend::Util::IsSupportedAVX || !TensorShaderAvxBackend::Util::IsSupportedAVX2) {
        throw gcnew System::PlatformNotSupportedException("AVX2 not supported on this platform.");
    }
}

TensorShaderAvxBackend::Trimming::Trimming() {
    if (!TensorShaderAvxBackend::Util::IsSupportedAVX || !TensorShaderAvxBackend::Util::IsSupportedAVX2) {
        throw gcnew System::PlatformNotSupportedException("AVX2 not supported on this platform.");
    }
}

TensorShaderAvxBackend::Zoom::Zoom() {
    if (!TensorShaderAvxBackend::Util::IsSupportedAVX || !TensorShaderAvxBackend::Util::IsSupportedAVX2) {
        throw gcnew System::PlatformNotSupportedException("AVX2 not supported on this platform.");
    }
}

TensorShaderAvxBackend::Convolution::Convolution() {
    if (!TensorShaderAvxBackend::Util::IsSupportedAVX || !TensorShaderAvxBackend::Util::IsSupportedAVX2) {
        throw gcnew System::PlatformNotSupportedException("AVX2 not supported on this platform.");
    }
    if (!TensorShaderAvxBackend::Util::IsSupportedFMA) {
        throw gcnew System::PlatformNotSupportedException("FMA not supported on this platform.");
    }
}

TensorShaderAvxBackend::Complex::Complex() {
    if (!TensorShaderAvxBackend::Util::IsSupportedAVX || !TensorShaderAvxBackend::Util::IsSupportedAVX2) {
        throw gcnew System::PlatformNotSupportedException("AVX2 not supported on this platform.");
    }
    if (!TensorShaderAvxBackend::Util::IsSupportedFMA) {
        throw gcnew System::PlatformNotSupportedException("FMA not supported on this platform.");
    }
}

TensorShaderAvxBackend::Trivector::Trivector() {
    if (!TensorShaderAvxBackend::Util::IsSupportedAVX || !TensorShaderAvxBackend::Util::IsSupportedAVX2) {
        throw gcnew System::PlatformNotSupportedException("AVX2 not supported on this platform.");
    }
    if (!TensorShaderAvxBackend::Util::IsSupportedFMA) {
        throw gcnew System::PlatformNotSupportedException("FMA not supported on this platform.");
    }
}

TensorShaderAvxBackend::Quaternion::Quaternion() {
    if (!TensorShaderAvxBackend::Util::IsSupportedAVX || !TensorShaderAvxBackend::Util::IsSupportedAVX2) {
        throw gcnew System::PlatformNotSupportedException("AVX2 not supported on this platform.");
    }
    if (!TensorShaderAvxBackend::Util::IsSupportedFMA) {
        throw gcnew System::PlatformNotSupportedException("FMA not supported on this platform.");
    }
}