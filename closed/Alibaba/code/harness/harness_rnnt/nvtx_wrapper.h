/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// ================================
//     Debug support: nvtx ranges
// ================================
#pragma once
#include <nvtx3/nvToolsExt.h> // For NVTX annotations

#define NVTX_ON

#define RGB_COLOR(R,G,B)    ((R<<16) + (G<<8) + B)
#ifdef NVTX_ON
   enum nvtx_color  {
      COLOR_BLUE_0  = 255,     COLOR_BLUE_1  = 200,     COLOR_BLUE_2  = 150,     COLOR_BLUE_3  = 100,
      COLOR_GREEN_0 = 255<<8,  COLOR_GREEN_1 = 200<<8,  COLOR_GREEN_2 = 150<<8,  COLOR_GREEN_3 = 100<<8,
      COLOR_RED_0   = 255<<16, COLOR_RED_1   = 200<<16, COLOR_RED_2   = 150<<16, COLOR_RED_3   = 100<<16,
      COLOR_YELLOW_0 = RGB_COLOR(255,255,0), COLOR_YELLOW_1 = RGB_COLOR(189,183,107), COLOR_YELLOW_2 = RGB_COLOR(238,232,170), COLOR_YELLOW_3 = RGB_COLOR(255,250,205),
      COLOR_YELLOW_4 = RGB_COLOR(240,230,140), COLOR_YELLOW_5 = RGB_COLOR(255,255,224), COLOR_YELLOW_6 = RGB_COLOR(255,228,181), COLOR_YELLOW_7 = RGB_COLOR(255,218,185),
      COLOR_PINK_0  = RGB_COLOR(255,192,203), COLOR_PINK_1 = RGB_COLOR(255-30,192-30,203-30)
   };
   #define NVTX_START(A, B, C)   nvtxRangeId_t A = thread_event_start(B, C)
   #define NVTX_START_WITH_PAYLOAD(A, B, C, D)   A = thread_event_start(B, C, D)
   #define NVTX_END(A)           thread_event_end(A)
#else
   #define NVTX_START(A, B, C)
   #define NVTX_START_WITH_PAYLOAD(A, B, C, D)
   #define NVTX_END(a)
#endif


#ifdef NVTX_ON
nvtxRangeId_t thread_event_start(const std::string &eventName, const nvtx_color eventColor, int64_t payload = 0)
{
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = eventColor;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = eventName.data();
    eventAttrib.payloadType = NVTX_PAYLOAD_TYPE_INT64;
    eventAttrib.payload.ullValue = payload;

    nvtxRangeId_t corrId = nvtxRangeStartEx(&eventAttrib);
    return (corrId);
}

void  thread_event_end(nvtxRangeId_t corrId)
{
    nvtxRangeEnd(corrId);
}
#endif
