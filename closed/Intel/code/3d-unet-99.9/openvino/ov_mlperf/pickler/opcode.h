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

namespace pickler {

enum Opcode {
    PROTO = 0x80,

    EMPTY_LIST = 0x5D,
    BINPUT = 0x71,
    MARK = 0x28,
    BINUNICODE = 0x58,
    APPENDS = 0x65,
    GLOBAL = 0x63,
    BINBYTES = 0x42,
    BININT1 = 0x4B,
    TUPLE = 0x74,
    TUPLE1 = 0x85,
    TUPLE2 = 0x86,
    TUPLE3 = 0x87,
    NEWTRUE = 0x88,
    NEWFALSE = 0x89,
    SHORT_BINBYTES = 0x43,
    REDUCE = 0x52,
    NONE = 0x4E,
    BININT = 0x4A,
    BUILD = 0x62,
    STOP = 0x2E
};

}
