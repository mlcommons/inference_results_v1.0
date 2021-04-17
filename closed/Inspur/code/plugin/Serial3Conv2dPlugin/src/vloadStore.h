#pragma once
/* Vectorized load and store */
#include<cuda_pipeline.h>
template<typename IType, int loadBytes>
RES_DEVICE void load(IType* dst, void const *ptr) {
  *dst=*(reinterpret_cast<IType const*>(ptr));  
}

template<typename IType, int loadBytes>
RES_DEVICE void store(void *dst, IType const* ptr) {
  *(reinterpret_cast<IType *>(dst)) = *ptr;    
}

#define IType int8_t
template<>
RES_DEVICE void load<IType, 16>(IType* dst, void const *ptr) {
  uint4* data=reinterpret_cast<uint4 *>(dst);
  uint4 const* src=reinterpret_cast<uint4 const*>(ptr);
  (*data).x = (*src).x;
  (*data).y = (*src).y;
  (*data).z = (*src).z;
  (*data).w = (*src).w;
}

template<>
RES_DEVICE void load<IType, 32>(IType* dst, void const *ptr) {
  uint4* data=reinterpret_cast<uint4 *>(dst);
  uint4 const* src=reinterpret_cast<uint4 const*>(ptr);
  __pipeline_memcpy_async(&data[0], &src[0], sizeof(uint4));
}


 
template<>
RES_DEVICE void load<IType, 8>(IType* dst, void const *ptr) {
  uint2* data=reinterpret_cast<uint2 *>(dst);
  uint2 const* src=reinterpret_cast<uint2 const*>(ptr);
  (*data).x = (*src).x;
  (*data).y = (*src).y;  
} 

template<>
RES_DEVICE void load<IType, 4>(IType* dst, void const *ptr) {
  unsigned* data=reinterpret_cast<unsigned *>(dst);
  unsigned const* src=reinterpret_cast<unsigned const*>(ptr);  
  *data = *src;
}

template<>
RES_DEVICE void load<IType, 2>(IType* dst, void const *ptr) {
  uint16_t* data=reinterpret_cast<uint16_t *>(dst);
  uint16_t const* src=reinterpret_cast<uint16_t const*>(ptr);
  *data = *src;
}

template<>
RES_DEVICE void store<IType, 16>(void *dst, IType const* ptr) {
  uint4 const* src = reinterpret_cast<uint4 const *>(ptr);
  uint4* data = reinterpret_cast<uint4 *>(dst);
  (*data).x = (*src).x;
  (*data).y = (*src).y;
  (*data).z = (*src).z;
  (*data).w = (*src).w;    
}



template<>
RES_DEVICE void store<IType, 8>(void *dst, IType const* ptr) {
  uint2 const* src = reinterpret_cast<uint2 const *>(ptr);
  uint2* data = reinterpret_cast<uint2 *>(dst);
  (*data).x = (*src).x;
  (*data).y = (*src).y;
}

template<>
RES_DEVICE void store<IType, 4>(void *dst, IType const* ptr) {
  unsigned const* src = reinterpret_cast<unsigned const *>(ptr);
  unsigned* data = reinterpret_cast<unsigned *>(dst);
  *data = *src;
}

template<>
RES_DEVICE void store<IType, 2>(void *dst, IType const* ptr) {
  uint16_t const* src = reinterpret_cast<uint16_t const *>(ptr);
  uint16_t* data = reinterpret_cast<uint16_t *>(dst);
  *data = *src;
}


 
 
 
 

