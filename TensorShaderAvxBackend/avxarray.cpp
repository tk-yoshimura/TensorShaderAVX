#include "TensorShaderAvxBackend.h"

#include <stdlib.h>
#include <memory.h>

using namespace System::Runtime::InteropServices;

generic <typename T>
IntPtr TensorShaderAvxBackend::AvxArray<T>::Ptr::get() {
    if (ptr == IntPtr::Zero) {
        throw gcnew System::InvalidOperationException();
    }

    return ptr;
}

generic <typename T>
UInt64 TensorShaderAvxBackend::AvxArray<T>::Length::get() {
    return length;
}

generic <typename T>
UInt64 TensorShaderAvxBackend::AvxArray<T>::MaxLength::get() {
    return MaxByteSize / ElementSize;
}

generic <typename T>
UInt64 TensorShaderAvxBackend::AvxArray<T>::ByteSize::get() {
    return ElementSize * Length;
}

generic <typename T>
UInt64 TensorShaderAvxBackend::AvxArray<T>::MaxByteSize::get() {
    return 0x40000000ull;
}

generic <typename T>
UInt64 TensorShaderAvxBackend::AvxArray<T>::ElementSize::get() {
    return sizeof(T);
}

generic <typename T>
UInt64 TensorShaderAvxBackend::AvxArray<T>::Alignment::get() {
    return 32;
}

generic <typename T>
bool TensorShaderAvxBackend::AvxArray<T>::IsValid::get() {
    return ptr != IntPtr::Zero;
}

generic <typename T>
Type^ TensorShaderAvxBackend::AvxArray<T>::ElementType::get() {
    return T::typeid;
}

generic <typename T>
cli::array<T>^ TensorShaderAvxBackend::AvxArray<T>::Value::get(){
    return (cli::array<T>^)this;
}


generic <typename T>
String^ TensorShaderAvxBackend::AvxArray<T>::Overview::get() {
    return ElementType->Name + "[" + Length + "]";
}

generic <typename T>
TensorShaderAvxBackend::AvxArray<T>::AvxArray(UInt64 length) {
    if (length > MaxLength) {
        throw gcnew System::ArgumentOutOfRangeException("length");
    }

    size_t size = ((length * ElementSize + Alignment - 1) / Alignment) * Alignment;

    void* ptr = _aligned_malloc(size, Alignment);
    if (ptr == nullptr) {
        throw gcnew System::OutOfMemoryException();
    }

    this->length = length;
    this->ptr = IntPtr(ptr);

    Zeroset();
}

generic <typename T>
TensorShaderAvxBackend::AvxArray<T>::AvxArray(cli::array<T>^ array) {
    UInt64 length = (UInt64)array->LongLength;

    if (length > MaxLength) {
        throw gcnew System::ArgumentOutOfRangeException("length");
    }

    size_t size = ((length * ElementSize + Alignment - 1) / Alignment) * Alignment;

    void* ptr = _aligned_malloc(size, Alignment);
    if (ptr == nullptr) {
        throw gcnew System::OutOfMemoryException();
    }

    this->length = length;
    this->ptr = IntPtr(ptr);

    Write(array);
}

generic <typename T>
T TensorShaderAvxBackend::AvxArray<T>::default::get(UInt64 index){
    if (index >= length) {
        throw gcnew System::IndexOutOfRangeException();
    }

    void* ptr = (void*)((UInt64)Ptr.ToInt64() + ElementSize * index);

    cli::array<T> ^buffer = gcnew cli::array<T>(1);

    pin_ptr<T> buffer_ptr = &(buffer[0]);

    memcpy_s((void*)buffer_ptr, ElementSize, ptr, ElementSize);

    return buffer[0];
}

generic <typename T>
void TensorShaderAvxBackend::AvxArray<T>::default::set(UInt64 index, T value) {
    if (index >= length) {
        throw gcnew System::IndexOutOfRangeException();
    }

    void* ptr = (void*)((UInt64)Ptr.ToInt64() + ElementSize * index);

    cli::array<T>^ buffer = gcnew cli::array<T>(1) { value };

    pin_ptr<T> buffer_ptr = &(buffer[0]);

    memcpy_s(ptr, ElementSize, (void*)buffer_ptr, ElementSize);
}

generic <typename T>
TensorShaderAvxBackend::AvxArray<T>::operator AvxArray^(cli::array<T>^ array) {
    return gcnew AvxArray(array);
}

generic <typename T>
TensorShaderAvxBackend::AvxArray<T>::operator cli::array<T>^ (AvxArray^ array) {
    if (array->length > (UInt64)int::MaxValue) {
        throw gcnew System::ArgumentOutOfRangeException("length");
    }

    cli::array<T>^ arr = gcnew cli::array<T>((int)array->length);
    array->Read(arr);

    return arr;
}

generic <typename T>
void TensorShaderAvxBackend::AvxArray<T>::Write(cli::array<T>^ array){
    Write(array, length);
}

generic <typename T>
void TensorShaderAvxBackend::AvxArray<T>::Read(cli::array<T>^ array) {
    Read(array, length);
}

generic <typename T>
void TensorShaderAvxBackend::AvxArray<T>::Write(cli::array<T>^ array, UInt64 count) {
    if (count > length || count > (UInt64)array->LongLength) {
        throw gcnew System::ArgumentOutOfRangeException("count");
    }

    if (count <= 0) {
        return;
    }

    void* ptr = Ptr.ToPointer();

    pin_ptr<T> array_ptr = &(array[0]);

    memcpy_s(ptr, count * ElementSize, (void*)array_ptr, count * ElementSize);
}

generic <typename T>
void TensorShaderAvxBackend::AvxArray<T>::Read(cli::array<T>^ array, UInt64 count) {
    if (count > length || count > (UInt64)array->LongLength) {
        throw gcnew System::ArgumentOutOfRangeException("count");
    }

    if (count <= 0) {
        return;
    }

    void* ptr = Ptr.ToPointer();

    pin_ptr<T> array_ptr = &(array[0]);

    memcpy_s((void*)array_ptr, count * ElementSize, ptr, count * ElementSize);
}

generic <typename T>
void TensorShaderAvxBackend::AvxArray<T>::Zeroset() {
    Zeroset(length);
}

generic <typename T>
void TensorShaderAvxBackend::AvxArray<T>::Zeroset(UInt64 count) {
    Zeroset(0, count);
}

generic <typename T>
void TensorShaderAvxBackend::AvxArray<T>::Zeroset(UInt64 index, UInt64 count) {
    void* ptr = (void*)((UInt64)Ptr.ToInt64() + ElementSize * index);

    memset(ptr, 0, count * ElementSize);
}

generic <typename T>
void TensorShaderAvxBackend::AvxArray<T>::CopyTo(AvxArray^ array, UInt64 count) {
    Copy(this, array, count);
}

generic <typename T>
void TensorShaderAvxBackend::AvxArray<T>::CopyTo(UInt64 index, AvxArray^ dst_array, UInt64 dst_index, UInt64 count) {
    Copy(this, index, dst_array, dst_index, count);
}

generic <typename T>
void TensorShaderAvxBackend::AvxArray<T>::Copy(AvxArray^ src_array, AvxArray^ dst_array, UInt64 count) {
    if (count > src_array->Length || count > dst_array->Length) {
        throw gcnew System::ArgumentOutOfRangeException("count");
    }

    void* src_ptr = src_array->Ptr.ToPointer();
    void* dst_ptr = dst_array->Ptr.ToPointer();

    memcpy_s(dst_ptr, count * ElementSize, src_ptr, count * ElementSize);
}

generic <typename T>
void TensorShaderAvxBackend::AvxArray<T>::Copy(AvxArray^ src_array, UInt64 src_index, AvxArray^ dst_array, UInt64 dst_index, UInt64 count) {
    if (src_index >= src_array->Length || src_index + count > src_array->Length) {
        throw gcnew ArgumentOutOfRangeException("src_index");
    }
    if (dst_index >= dst_array->Length || dst_index + count > dst_array->Length) {
        throw gcnew ArgumentOutOfRangeException("dst_index");
    }

    void* src_ptr = (void*)((UInt64)src_array->Ptr.ToInt64() + ElementSize * src_index);
    void* dst_ptr = (void*)((UInt64)dst_array->Ptr.ToInt64() + ElementSize * dst_index);

    memcpy_s(dst_ptr, count * ElementSize, src_ptr, count * ElementSize);
}


generic <typename T>
String^ TensorShaderAvxBackend::AvxArray<T>::ToString() {
    return "AvxArray " + Overview;
}

generic <typename T>
TensorShaderAvxBackend::AvxArray<T>::~AvxArray() {
    this->!AvxArray();
}

generic <typename T>
TensorShaderAvxBackend::AvxArray<T>::!AvxArray() {
    if (this->ptr != IntPtr::Zero) {
        void* ptr = (this->ptr).ToPointer();

        _aligned_free(ptr);

        this->length = 0;
        this->ptr = IntPtr::Zero;
    }
}