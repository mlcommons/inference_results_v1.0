#pragma once
#include "macro.h"
#include "vloadStore.h"
/* LDG */
// Load Inputs from global memory to shared memory.


RES_DEVICE void LDG(int8_t* data, int8_t* weight, int8_t* A_SHARE, int8_t* B_SHARE, int koffset = 0) {

	int tid = threadIdx.x;
	int tidInWarp = tid % 32;
	int wid = tid / 32;

	int bx = blockIdx.x;  //OC
	int by = blockIdx.y;  //7*7
	int bz = blockIdx.z;  //N/8

	int tidInRowPerWarp = tidInWarp / 2;
	int tidInColPerWarp = tidInWarp % 2;
	load<int8_t, 16>((A_SHARE  //+ (koffset / 64) * (64 * 4 * 4 * 4 + 128 / 4 * 16)      //*4*4*4*64
		+ wid % 2 * (4 * 4 * 4 * 32 + 64 / 4 * 16)      //*4*4*8*32
		+ wid / 2 * (512 + 16 / 4 * 16)       // *4*4*32
		+ tidInRowPerWarp * 32
		+ (tidInRowPerWarp / 4) * 16
		+ tidInColPerWarp * 16
		)
		, (void*)(data
			+ bz * 401408 //*4*28*28*128
			+ by * 16 * 32   //*4*28*32
			+ wid / 2 * 100352    //*128*28*28
			+ (wid % 2 * 32 + 0*0) * 784 //*28*28  //---------
			+ tidInRowPerWarp * 32
			+ tidInColPerWarp * 16
			));  // VEC len kVEC 


	load<int8_t, 16>((B_SHARE
		//+ (koffset / 64) % 2 * (8192 + 256 / 4 * 16) //*64*128
		+ tid / 2 * 32
		+ tid % 2 * 16
		+ tid / 8 * 16
		)
		, (void*)(weight//+ koffset * 128
			+ bx * 16384 //*128*128
			+ tid / 2 * 32
			+ tid % 2 * 16
			));  // VEC len kVEC

	load<int8_t, 16>((B_SHARE
		//+ (koffset / 64) % 2 * (8192 + 256 / 4 * 16)//*64*128
		+ tid / 2 * 32
		+ tid % 2 * 16
		+ tid / 8 * 16
		+ (32 * 128 + 128 / 4 * 16)
		)
		, (void*)(weight  //+ koffset * 128  
			+ bx * 16384 //*128*128
			+ tid / 2 * 32
			+ tid % 2 * 16
			+ 32 * 128
			));  // VEC len kVEC


	/*load<int8_t, 16>((A_SHARE
		+ (koffset / 64) * (64 * 4 * 4 * 4 + 128 / 4 * 16)      //*4*4*4*64
		+ wid % 2 * (4 * 4 * 4 * 32 + 64 / 4 * 16)      //*4*4*8*32
		+ wid / 2 * (512 + 16 / 4 * 16)       // *4*4*32
		+ tidInRowPerWarp * 32
		+ (tidInRowPerWarp / 4) * 16
		+ tidInColPerWarp * 16
		)
		, (void*)(data
			+ bz * 401408 //*4*28*28*128
			+ by * 16 * 32   //*4*28*32
			+ wid / 2 * 100352    //*128*28*28
			+ (wid % 2 * 32 + koffset) * 784 //*28*28
			+ tidInRowPerWarp * 32
			+ tidInColPerWarp * 16
			));  // VEC len kVEC 


	load<int8_t, 16>((B_SHARE
		+ (koffset / 64) % 2 * (8192 + 256 / 4 * 16) //*64*128
		+ tid / 2 * 32
		+ tid % 2 * 16
		+ tid / 8 * 16
		)
		, (void*)(weight
			+ koffset * 128
			+ bx * 16384 //*128*128
			+ tid / 2 * 32
			+ tid % 2 * 16
			));  // VEC len kVEC

	load<int8_t, 16>((B_SHARE
		+ (koffset / 64) % 2 * (8192 + 256 / 4 * 16)//*64*128
		+ tid / 2 * 32
		+ tid % 2 * 16
		+ tid / 8 * 16
		+ (32 * 128 + 128 / 4 * 16)
		)
		, (void*)(weight
			+ koffset * 128
			+ bx * 16384 //*128*128
			+ tid / 2 * 32
			+ tid % 2 * 16
			+ 32 * 128
			));  // VEC len kVEC  */

   /* load<int8_t,16>((B_SHARE
			+ (koffset/64)%2      *8192 // *64*128
			+	tid/4               *64
			+ tid%4	            *16
			)
		  , (void*) (weight
			+ koffset             *512
			+ bx                  *8192 // *128*64
			+ tid/4               *64
			+ tid%4	            *16
			));  // VEC len kVEC

	  load<int8_t,16>((B_SHARE
			+ (koffset/64)%2      *8192 // *64*128
			+	tid/4               *64
			+ tid%4	            *16
			+ 4096 //64*64
			)
		  , (void*) (weight
			+ koffset             *512         //0鎴?4
			+ bx                  *8192 // *128*64
			+ tid/4               *64
			+ tid%4	            *16
			+ 64*64
			));  // */

}


RES_DEVICE void STG(int8_t* output, int8_t* A_SHARE) {

	int tid = threadIdx.x;
	int tidInWarp = tid % 32;
	int wid = tid / 32;

	int bx = blockIdx.x;  //OC
	int by = blockIdx.y;  //7*7
	int bz = blockIdx.z;  //N/8

	int tidInRowPerWarp = tidInWarp / 2;
	int tidInColPerWarp = tidInWarp % 2;

	for (int i = 0; i < 2; i++) {
		store<int8_t, 16>((void*)(output
			+ bx * 100352 //*28*28*128
			+ bz * 4 * 28 * 28 * 512 //*4*28*28*512
			+ by * 16 * 32
			+ (wid / 4 + i * 2) * 401408   //*512*28*28
			+ wid % 4 * 25088 //*32*28*28
			+ tidInRowPerWarp * 32
			+ tidInColPerWarp * 16
			),
			(A_SHARE
				+ wid % 4 * 4 * 4 * 4 * 32 //*4*4*8*32
				+ (wid / 4 + i * 2) * 512  // *4*4*32
				+ tidInRowPerWarp * 32
				+ tidInColPerWarp * 16
				));  // VEC len kVEC 
	}
}



RES_DEVICE void LDGA(int8_t* A_SHARE, int8_t* A_DATA) {

	int tid = threadIdx.x;
	int tidInWarp = tid % 32;
	int wid = tid / 32;

	int bx = blockIdx.x;  //OC
	int by = blockIdx.y;  //7*7
	int bz = blockIdx.z;  //N/8

	int tidInRowPerWarp = tidInWarp / 2;
	int tidInColPerWarp = tidInWarp % 2;
	
	for (int i = 0; i < 2; i++) {
		load<int8_t, 32>((A_SHARE
			+ wid % 4 * 4 * 4 * 4 * 32 //*4*4*8*32
			+ (wid / 4 + i * 2) * 512  // *4*4*32
			+ tidInRowPerWarp * 32
			+ tidInColPerWarp * 16
			),
			(void*)(A_DATA
				+ bx * 100352 //*28*28*128
				+ bz * 4 * 28 * 28 * 512 //*4*28*28*512
				+ by * 16 * 32
				+ (wid / 4 + i * 2) * 401408   //*512*28*28
				+ wid % 4 * 25088 //*32*28*28
				+ tidInRowPerWarp * 32
				+ tidInColPerWarp * 16
				));  // VEC len kVEC

	}
	__pipeline_commit();
}

RES_DEVICE void LDGScale(float* dst_scale, float* src_scale, float* dst_bias, float* src_bias) {
	int tid = threadIdx.x;
	int bx = blockIdx.x;
	if (tid < 128) {
		dst_scale[tid] = src_scale[tid + bx * 128];
		dst_bias[tid] = src_bias[tid + bx * 128];
	}
}

RES_DEVICE void LDG1(int8_t* data, int8_t* weight, int8_t* A_SHARE, int8_t* B_SHARE, int koffset, const int batch_size) {

	int tid = threadIdx.x;
	int tidInWarp = tid % 32;
	int wid = tid / 32;

	int bx = blockIdx.x;  //OC
	int by = blockIdx.y;  //7*7
	int bz = blockIdx.z;  //N/8

	int tidInRowPerWarp = tidInWarp / 2;
	int tidInColPerWarp = tidInWarp % 2;
	int data_offset = bz * 401408 //*4*28*28*128
		+ by * 16 * 32   //*4*28*32
		+ wid / 2 * 100352    //*128*28*28
		+ (wid % 2 * 32 + 0) * 784 //*28*28
		+ tidInRowPerWarp * 32
		+ tidInColPerWarp * 16;
	if (data_offset < batch_size * 28 * 28 * 128)
		load<int8_t, 16>((A_SHARE
			//+ (koffset / 64) * (64 * 4 * 4 * 4 + 128 / 4 * 16)      //*4*4*4*64
			+ wid % 2 * (4 * 4 * 4 * 32 + 64 / 4 * 16)      //*4*4*8*32
			+ wid / 2 * (512 + 16 / 4 * 16)       // *4*4*32
			+ tidInRowPerWarp * 32
			+ (tidInRowPerWarp / 4) * 16
			+ tidInColPerWarp * 16
			)
			, (void*)(data + data_offset
				));  // VEC len kVEC 


	load<int8_t, 16>((B_SHARE
		//+ (koffset / 64) % 2 * (8192 + 256 / 4 * 16) //*64*128
		+ tid / 2 * 32
		+ tid % 2 * 16
		+ tid / 8 * 16
		)
		, (void*)(weight
			//+ koffset * 128
			+ bx * 16384 //*128*128
			+ tid / 2 * 32
			+ tid % 2 * 16
			));  // VEC len kVEC

	load<int8_t, 16>((B_SHARE
		//+ (koffset / 64) % 2 * (8192 + 256 / 4 * 16)//*64*128
		+ tid / 2 * 32
		+ tid % 2 * 16
		+ tid / 8 * 16
		+ (32 * 128 + 128 / 4 * 16)
		)
		, (void*)(weight
			//+ koffset * 128
			+ bx * 16384 //*128*128
			+ tid / 2 * 32
			+ tid % 2 * 16
			+ 32 * 128
			));  // VEC len kVEC

   /* load<int8_t,16>((B_SHARE
			+ (koffset/64)%2      *8192 // *64*128
			+	tid/4               *64
			+ tid%4	            *16
			)
		  , (void*) (weight
			+ koffset             *512
			+ bx                  *8192 // *128*64
			+ tid/4               *64
			+ tid%4	            *16
			));  // VEC len kVEC

	  load<int8_t,16>((B_SHARE
			+ (koffset/64)%2      *8192 // *64*128
			+	tid/4               *64
			+ tid%4	            *16
			+ 4096 //64*64
			)
		  , (void*) (weight
			+ koffset             *512         //0鎴?4
			+ bx                  *8192 // *128*64
			+ tid/4               *64
			+ tid%4	            *16
			+ 64*64
			));  // */

}


RES_DEVICE void STG1(int8_t* output, int8_t* A_SHARE, const int batch_size) {

	int tid = threadIdx.x;
	int tidInWarp = tid % 32;
	int wid = tid / 32;

	int bx = blockIdx.x;  //OC
	int by = blockIdx.y;  //7*7
	int bz = blockIdx.z;  //N/8

	int tidInRowPerWarp = tidInWarp / 2;
	int tidInColPerWarp = tidInWarp % 2;

	for (int i = 0; i < 2; i++) {
		int data_offset = bx * 100352 //*28*28*128
			+ bz * 4 * 28 * 28 * 512 //*4*28*28*512
			+ by * 16 * 32
			+ (wid / 4 + i * 2) * 401408   //*512*28*28
			+ wid % 4 * 25088 //*32*28*28
			+ tidInRowPerWarp * 32
			+ tidInColPerWarp * 16;
		if (data_offset < batch_size * 28 * 28 * 512)
			store<int8_t, 16>((void*)(output + data_offset
				),
				(A_SHARE
					+ wid % 4 * 4 * 4 * 4 * 32 //*4*4*8*32
					+ (wid / 4 + i * 2) * 512  // *4*4*32
					+ tidInRowPerWarp * 32
					+ tidInColPerWarp * 16
					));  // VEC len kVEC 
	}
}



RES_DEVICE void LDGA1(int8_t* A_SHARE, int8_t* A_DATA, const int batch_size) {

	int tid = threadIdx.x;
	int tidInWarp = tid % 32;
	int wid = tid / 32;

	int bx = blockIdx.x;  //OC
	int by = blockIdx.y;  //7*7
	int bz = blockIdx.z;  //N/8

	int tidInRowPerWarp = tidInWarp / 2;
	int tidInColPerWarp = tidInWarp % 2;

	for (int i = 0; i < 2; i++) {
		int data_offset = bx * 100352 //*28*28*128
			+ bz * 4 * 28 * 28 * 512 //*4*28*28*512
			+ by * 16 * 32
			+ (wid / 4 + i * 2) * 401408   //*512*28*28
			+ wid % 4 * 25088 //*32*28*28
			+ tidInRowPerWarp * 32
			+ tidInColPerWarp * 16;
		if (data_offset < batch_size * 28 * 28 * 512)
			load<int8_t, 32>((A_SHARE
				+ wid % 4 * 4 * 4 * 4 * 32 //*4*4*8*32
				+ (wid / 4 + i * 2) * 512  // *4*4*32
				+ tidInRowPerWarp * 32
				+ tidInColPerWarp * 16
				),
				(void*)(A_DATA + data_offset
					));  // VEC len kVEC

	}
	__pipeline_commit();
}



