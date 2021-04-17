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

#include <string>
#include <vector>
#include <fstream>

#include "opcode.h"

namespace pickler {

class Parser {
public:
    static std::vector<std::string> list(const std::string& path) {
        std::vector<std::string> result;

        std::ifstream stream(path, std::ios::binary);

        if (!stream.is_open()) {
            throw "Can not open pickle file.";
        }

        uint8_t opcode = 0;
        char* opcode_ptr = reinterpret_cast<char*>(&opcode);

        if (!checkVersion(stream)) {
            throw "Unsupported pickle version.";
        }

        while (opcode != Opcode::STOP) {
            stream.read(opcode_ptr, sizeof(uint8_t));

            switch (opcode) {
                case Opcode::EMPTY_LIST:
                    return parseList(stream);
                default:
                    throw "Unsupported pickle opcode.";
            }
        }

        return result;
    }

    static bool dataOffsetAndSize(const std::string& path, size_t& offset, size_t& size) {
        std::ifstream stream(path, std::ios::binary);

        uint8_t opcode = 0;
        char* opcode_ptr = reinterpret_cast<char*>(&opcode);

        uint8_t count = 0;
        char* count_ptr = reinterpret_cast<char*>(&count);

        uint8_t binint1 = 0;
        char* binint1_ptr = reinterpret_cast<char*>(&binint1);

        uint32_t binint = 0;
        char* binint_ptr = reinterpret_cast<char*>(&binint);

        uint8_t value = 0;
        char* value_ptr = reinterpret_cast<char*>(&value);

        std::string module_name;
        std::string class_name;

        if (!checkVersion(stream)) {
            throw "Unsupported pickle version.";
        }

        while (opcode != Opcode::STOP) {
            stream.read(opcode_ptr, sizeof(uint8_t));

            switch (opcode) {
                case Opcode::GLOBAL:
                    module_name = readString(stream);
                    class_name = readString(stream);
                    break;
                case Opcode::REDUCE:
                    break;
                case Opcode::EMPTY_LIST:
                    break;
                case Opcode::MARK:
                    break;
                case Opcode::BINPUT:
                    stream.read(binint_ptr, sizeof(uint8_t));
                    break;
                case Opcode::BININT:
                    stream.read(binint_ptr, sizeof(uint32_t));
                    break;
                case Opcode::BININT1:
                    stream.read(binint1_ptr, sizeof(uint8_t));
                    break;
                case Opcode::BINUNICODE:
                    readUnicodeString(stream);
                    break;
                case Opcode::NONE:
                    break;
                case Opcode::TUPLE:
                    break;
                case Opcode::TUPLE1:
                case Opcode::TUPLE2:
                case Opcode::TUPLE3:
                    break;
                case Opcode::NEWTRUE:
                case Opcode::NEWFALSE:
                    break;
                case Opcode::BUILD:
                    break;
                case Opcode::BINBYTES:
                    stream.read(binint_ptr, sizeof(uint32_t));
                    size = binint;
                    offset = stream.tellg();
                    return true;
                case Opcode::SHORT_BINBYTES:
                    stream.read(count_ptr, sizeof(uint8_t));
                    for (int i = 0; i < count; i++) {
                        stream.read(value_ptr, sizeof(uint8_t));
                    }
                    break;
                default:
                    throw "Unsupported pickle opcode.";
            }
        }

        return false;
    }

private:
    static bool checkVersion(std::ifstream& stream) {
        uint8_t opcode = 0;
        char* opcode_ptr = reinterpret_cast<char*>(&opcode);

        stream.read(opcode_ptr, sizeof(uint8_t));

        if (opcode != Opcode::PROTO) {
            return false;
        }

        stream.read(opcode_ptr, sizeof(uint8_t));

        if (opcode != 3) {
            return false;
        }

        return true;
    }

    static std::vector<std::string> parseList(std::ifstream& stream) {
        std::vector<std::string> result;

        uint32_t length;
        char* length_ptr = reinterpret_cast<char*>(&length);

        uint8_t opcode = 0;
        char* opcode_ptr = reinterpret_cast<char*>(&opcode);

        uint8_t count = 0;
        char* count_ptr = reinterpret_cast<char*>(&count);

        while (opcode != Opcode::STOP) {
            stream.read(opcode_ptr, sizeof(uint8_t));

            switch (opcode) {
                case Opcode::BINPUT:
                    stream.read(count_ptr, sizeof(uint8_t));
                    break;

                case Opcode::MARK:
                    break;

                case Opcode::BINUNICODE:
                    result.push_back(readUnicodeString(stream));
                    break;

                case Opcode::APPENDS:
                    break;

                case Opcode::STOP:
                    break;

                default:
                    throw "Unsupported pickle opcode.";
            }
        }

        return result;
    }

    static  std::string readString(std::ifstream& stream) {
        std::string result;
        char chr;
        stream.read(&chr, sizeof(chr));
        while (chr != '\n') {
            result.push_back(chr);
            stream.read(&chr, sizeof(chr));
        }

        return result;
    }

    static  std::string readUnicodeString(std::ifstream& stream) {
        std::string result;

        uint32_t length;
        char* length_ptr = reinterpret_cast<char*>(&length);

        stream.read(length_ptr, sizeof(uint32_t));

        char* buffer = new char[length + 1];

        stream.read(buffer, length);
        buffer[length] = '\0';

        result = buffer;\
        delete [] buffer;
        return result;
    }
};

}
