/*
 // Copyright (c) 2020 Intel Corporation
 //
 // Licensed under the Apache License, Version 2.0 (the "License");
 // you may not use this file except in compliance with the License.
 // You may obtain a copy of the License at
 //
 //      http://www.apache.org/licenses/LICENSE-2.0
 //
 // Unless required by applicable law or agreed to in writing, software
 // distributed under the License is distributed on an "AS IS" BASIS,
 // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 // See the License for the specific language governing permissions and
 // limitations under the License.
 */

#pragma once

#include <memory>

#ifdef _WIN32
#include <windows.h>
#define MAP_FAILED nullptr
#else
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#define INVALID_HANDLE_VALUE -1
#endif

class MmapAllocator {
#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    using path_type = std::wstring;
#else
    using path_type = std::string;
#endif

public:
    using Ptr = std::shared_ptr<MmapAllocator>;
    
    MmapAllocator(const path_type& path)
        : _path(path), _size(0), _data(nullptr) {
    }

    virtual ~MmapAllocator() {
        free();
    }

    void* alloc(size_t offset = 0, size_t size = 0) {
        free();
        map(_path.c_str(), offset, size);
        return data<void>();
    }

    template<typename T>
    T* data(size_t offset = 0) {
        return reinterpret_cast<T*>(_data + offset);
    }

    void free() {
        unmap();
        closeFile();
    }

    size_t size() {
        return _size;
    }

protected:
    void closeFile() {
        if (_file == INVALID_HANDLE_VALUE) {
            return;
        }
#ifdef _WIN32
        ::CloseHandle(_file);
#else
        close(_file);
#endif
        _file = INVALID_HANDLE_VALUE;
    }

    void map(const path_type& path, size_t offset = 0, size_t size = 0) {
#ifdef _WIN32
        SYSTEM_INFO SystemInfo;
        GetSystemInfo(&SystemInfo);
        const int64_t page_size = SystemInfo.dwAllocationGranularity;
#else
        const int64_t page_size = sysconf(_SC_PAGE_SIZE);
#endif
        const int64_t offset_align = offset;// / page_size * page_size;

        int64_t file_size;
#ifdef _WIN32
        DWORD file_mode = GENERIC_READ;
        DWORD map_mode = PAGE_READONLY;
        DWORD access = FILE_MAP_READ;

#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        _file = ::CreateFileW(path.c_str(), file_mode, FILE_SHARE_READ | FILE_SHARE_WRITE,
                             0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
#else
        _file = ::CreateFileA(path.c_str(), file_mode, FILE_SHARE_READ | FILE_SHARE_WRITE,
            0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
#endif

        if (_file == INVALID_HANDLE_VALUE) {
            THROW_IE_EXCEPTION << "Can not open file for mapping.";
        }

        LARGE_INTEGER file_size_large;
        if (::GetFileSizeEx(_file, &file_size_large) == 0) {
            closeFile();
            THROW_IE_EXCEPTION << "Can not get file size.";
        }

        file_size = static_cast<int64_t>(file_size_large.QuadPart);

        const int64_t map_size = offset - offset_align + (size == 0 ? file_size : size);

        const int64_t total_file_size = offset + map_size;

        if (total_file_size > file_size) {
            closeFile();
            THROW_IE_EXCEPTION << "File size is less than requested map size.";
        }

        _mapping = ::CreateFileMapping(_file, 0, access,
            total_file_size >> 32,
            total_file_size & 0xffffffff,
            0);

        if (_mapping == INVALID_HANDLE_VALUE) {
            closeFile();
            THROW_IE_EXCEPTION << "Can not create  file mapping.";
        }

        _data = ::MapViewOfFile(
            _mapping,
            map_mode,
            offset_align >> 32,
            offset_align & 0xffffffff,
            map_size);
#else
        int prot = PROT_READ;
        int mode = O_RDONLY;

        struct stat sb = {};
        _file = open(path.c_str(), mode);

        if (_file == INVALID_HANDLE_VALUE) {
            throw "Can not open file for mapping.";
        }

        if (fstat(_file, &sb) == -1) {
            closeFile();
            throw "Can not get file size.";
        }

        file_size = (size_t)sb.st_size;

        const int64_t map_size = offset - offset_align + (size == 0 ? file_size : size);

        const int64_t total_file_size = offset + map_size;

        if (total_file_size > file_size) {
            closeFile();
            throw "File size is less than requested map size.";
        }

        _data = reinterpret_cast<char*>(mmap(NULL, map_size, prot, MAP_PRIVATE, _file, offset_align));
    #endif
        if (_data == MAP_FAILED) {
            closeFile();
            throw "Can not create file mapping.";
        }

        _size = map_size;
    }

    void unmap() {
        if (_data != nullptr) {
#ifdef _WIN32
            ::UnmapViewOfFile(_data);
            ::CloseHandle(_mapping);
            _mapping = INVALID_HANDLE_VALUE;
#else
            munmap(_data, _size);
#endif
        }
        _data = nullptr;
        _size = 0;
    }

private:
    path_type _path;
    size_t _size;

    char* _data;
#ifdef _WIN32
    HANDLE _file = INVALID_HANDLE_VALUE;
    HANDLE _mapping = INVALID_HANDLE_VALUE;
#else
    int _file = INVALID_HANDLE_VALUE;
#endif
};
