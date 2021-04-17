#include "LDG.h"





__device__ void index_AB_SHARE(int* index_S) {

	int tid = threadIdx.x;
	int lane = tid & 31;
	//int warp = tid / 32;
	int num_col = 128 + 16;

	/*index_S[0] = lane / 16 * 16 + lane / 4 % 4 *  num_col
						+ lane % 4 * 32;
	index_S[1] = lane / 16 * 2 * num_col 
						+ lane / 4 % 4 / 2 * 16 + lane / 4 % 4 % 2 *  num_col
						+ lane % 4 * 32;  */
	index_S[0] = (lane >> 4 << 4) + (((lane >>2) & 3) + 0) *  num_col
						+ ((lane &3 ) << 5);
	index_S[1] = (lane >> 4 <<1) * num_col 
						+ (((lane >>2) & 3) >> 1 << 4) + ((((lane >> 2) & 3) & 1) + 0)*  num_col
						+ ((lane & 3) << 5);						
	//return index_S;
}


#if (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2)
extern "C" {
	//
	// This NVVM intrinsic is subject to change in future versions of CUDA.
	// Clients should not call it directly. Rather, they should use the
	// cutlass::arch::ldsm<>() template.
	//
	__device__ uint32_t __nvvm_get_smem_pointer(void*);
}
#endif



/// CUTLASS helper to get SMEM pointer
inline __device__ unsigned cutlass_get_smem_pointer(void *ptr) {

// We prefer to use the new CVTA intrinsics if they are available, otherwise we will fall back to
// the previous internal intrinsics if they are available.
#if (defined(__CUDA_ARCH__) && __CUDACC_VER_MAJOR__ >= 11)
  //
  // This NVVM intrinsic converts an address in shared memory to a plain
  // unsigned integer. This is necessary to pass to shared memory instructions
  // in inline PTX.
  //
  // In CUDA 11 and beyond, this replaces __nvvm_get_smem_pointer()  [only available in 10.2].
  //
  //__device__ size_t __cvta_generic_to_shared(void* ptr);

  /// CUTLASS helper to get SMEM pointer
  return static_cast<unsigned>(__cvta_generic_to_shared(ptr));

#elif (defined(__CUDA_ARCH__) &&  __CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2)

  return __nvvm_get_smem_pointer(ptr);

#elif defined(__CUDA_ARCH__)

  uint32_t smem_ptr;

  asm(
  "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
    : "=r"(smem_ptr) : "l"(ptr));

  return smem_ptr;

#else

  return 0;
#endif
}

/// CUTLASS helper to get SMEM pointer
inline __device__ unsigned cutlass_get_smem_pointer(void const *ptr) {
  return cutlass_get_smem_pointer(const_cast<void *>(ptr));
}

__device__ void LDS_AB_1(int8_t* A_SHARE, int8_t* B_SHARE, int8_t* A, int8_t* B, int* index_S, int iK, int ik) {
	int tid = threadIdx.x;
	int lane = tid & 31;
	int warp = tid >> 5;
	int num_col = 128 + 16;

	//int inner_shareA1 = (iK * 32 + ik * 16 + warp % 2 * 8) * num_col + index_S[0];
	//int inner_shareB1 = (iK * 64 + ik * 32 + warp / 2 * 8) * num_col + index_S[1];

	int inner_shareA1 = (warp % 2 * 8) * num_col + index_S[0];
	int inner_shareB1 = (warp / 2 * 8) * num_col + index_S[1];

	//int8_t B3[32];

	int x, y, z, w;
	
	void* ptr_A1 =
		reinterpret_cast<void*>(&A_SHARE[inner_shareA1]);
	unsigned addr_A1 = cutlass_get_smem_pointer(ptr_A1);
	asm volatile ("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(addr_A1));
	reinterpret_cast<int4&>(A[ik * 32]) = make_int4(x, y, z, w);
	//reinterpret_cast<int4&>(A[0]) = make_int4(x, y, z, w);		

	ptr_A1 =
		reinterpret_cast<void*>(&A_SHARE[inner_shareA1 + 4 * num_col]);
	addr_A1 = cutlass_get_smem_pointer(ptr_A1);
	asm volatile ("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(addr_A1));
	//reinterpret_cast<int4&>(A[16]) = make_int4(x, y, z, w);
	reinterpret_cast<int4&>(A[16 + ik * 32]) = make_int4(x, y, z, w);  // 
	
	
	void* ptr_B1 =
		reinterpret_cast<void*>(&B_SHARE[inner_shareB1]);
	unsigned addr_B1 = cutlass_get_smem_pointer(ptr_B1);
	asm volatile ("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(addr_B1));
	//reinterpret_cast<int4&>(B[0]) = make_int4(x, y, z, w);	
	reinterpret_cast<int4&>(B[ik * 32]) = make_int4(x, y, z, w);	

	ptr_B1 =
		reinterpret_cast<void*>(&B_SHARE[inner_shareB1 + 4 * num_col]);
	addr_B1 = cutlass_get_smem_pointer(ptr_B1);
	asm volatile ("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(addr_B1));
	//reinterpret_cast<int4&>(B[16]) = make_int4(x, y, z, w);
	reinterpret_cast<int4&>(B[16 + ik * 32]) = make_int4(x, y, z, w);  // */

}






/*__device__ void LDS_AB(int8_t* A_SHARE, int8_t* B_SHARE, int8_t* A, int8_t* B, int* index_S, int iK, int ik) {
	int tid = threadIdx.x;
	int warp = tid / 32;
	int num_col = 128 + 16;

	for (int i = 0; i < 2; i++) {  
		int inner_shareA = (iK * 32 + ik * 16 + warp % 2 * 8 + i * 4) * num_col + index_S[0];

		for (int j = 0; j < 2; j++) {
			for (int k = 0; k < 2; k++) {

				reinterpret_cast<int*>(&A[i * 16 + j * 8 + k * 4 + ik * 32])[0] = reinterpret_cast<int const*>(&A_SHARE[inner_shareA + k * 2 * num_col + j * 16])[0];
			}  //end for k
		}  //end for j
	}  //end for i

	for (int i = 0; i < 4; i++) {  

		int inner_shareB = (iK * 64 + ik * 32 + warp / 2 * 8 + i * 2) * (num_col)+index_S[0];

		for (int j = 0; j < 2; j++) {
			reinterpret_cast<int*>(&B[i * 8 + j * 4 + ik * 32])[0] = reinterpret_cast<int const*>(&B_SHARE[inner_shareB + j * 16])[0];
		}  //end for j
	}  //end for i

}  //  */

__device__ void MMA(int8_t* a, int8_t* b, int* acc) {

	for (int j = 0; j < 4; j++) {  
		for (int i = 0; i < 2; i++) {  

			uint32_t const* A = reinterpret_cast<uint32_t const*>(&a[i * 16]);
			uint32_t const* B = reinterpret_cast<uint32_t const*>(&b[j * 8]);
			int const* C = reinterpret_cast<int const*>(&acc[(j * 2 + i) * 4]);
			int* D = reinterpret_cast<int*>(&acc[((j * 2 + i)) * 4]);

			asm volatile(
				"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
				"{%8,%9}, {%10,%11,%12,%13};\n"
				: "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
				: "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
				"r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));

		}  //end for j
	}  //end for i

}



__device__ void Pipline_ldmat(int8_t* data, int8_t* weight, int8_t* A_SHARE, int8_t* B_SHARE, int* Acc) {
	int8_t A[64];  
	int8_t B[64];

	const int num_col = 128 + 16;
        //int8_t B2[64];	

	int index_S[2];
	index_AB_SHARE(index_S);
	
	int8_t* ldg_A_SHARE = A_SHARE;
	int8_t* ldg_B_SHARE = B_SHARE;

	LDG(data, weight, ldg_A_SHARE, ldg_B_SHARE, 0);  
	data += 64 * 784;
	weight += 64 * 128;
	ldg_A_SHARE += (64 * 4 * 4 * 4 + 128 / 4 * 16);
	ldg_B_SHARE += (8192 + 256 / 4 * 16);
	LDG(data, weight, ldg_A_SHARE, ldg_B_SHARE, 64);
	__syncthreads();


	LDS_AB_1(A_SHARE, B_SHARE, A, B, index_S, 0, 0);
	A_SHARE +=  num_col * 16;
	B_SHARE +=  num_col * 32;

	//LDS_AB(A_SHARE, B_SHARE, A, B, index_S, 0, 0);
	LDS_AB_1(A_SHARE, B_SHARE, A, B, index_S, 0, 1);
	A_SHARE =  A_SHARE - num_col * 16 + num_col * 32;
	B_SHARE =  B_SHARE - num_col * 32 + num_col * 64;	



	MMA(&A[0], &B[0], Acc);
	MMA(&A[32], &B[32], Acc);
	__syncthreads();

	LDS_AB_1(A_SHARE, B_SHARE, A, B, index_S, 1, 0);
	A_SHARE +=  num_col * 16;
	B_SHARE +=  num_col * 32;		

	
	LDS_AB_1(A_SHARE, B_SHARE, A, B, index_S, 1, 1);

	MMA(&A[0], &B[0], Acc);
	MMA(&A[32], &B[32], Acc);

}

/*__device__ void Pipline(int8_t* data, int8_t* weight, int8_t* A_SHARE, int8_t* B_SHARE, int* Acc) {
	int8_t A[64];  
	int8_t B[64];
        //int8_t B2[64];	

	int index_S[2];
	index_AB_SHARE(index_S);

	LDG(data, weight, A_SHARE, B_SHARE, 0);  
	LDG(data, weight, A_SHARE, B_SHARE, 64);
	__syncthreads();


	LDS_AB(A_SHARE, B_SHARE, A, B, index_S, 0, 0);
	//LDS_AB(A_SHARE, B_SHARE, A, B, index_S, 0, 0);
	LDS_AB(A_SHARE, B_SHARE, A, B, index_S, 0, 1);


	MMA(&A[0], &B[0], Acc);
	MMA(&A[32], &B[32], Acc);
	__syncthreads();

	LDS_AB(A_SHARE, B_SHARE, A, B, index_S, 1, 0);
	LDS_AB(A_SHARE, B_SHARE, A, B, index_S, 1, 1);

	MMA(&A[0], &B[0], Acc);
	MMA(&A[32], &B[32], Acc);

}*/

__device__ void Pipline1(int8_t* data, int8_t* weight, int8_t* A_SHARE, int8_t* B_SHARE, int* Acc, const int batch_size) {
	int8_t A[64];  
	int8_t B[64];  

	const int num_col = 128 + 16;
	int index_S[2];
	index_AB_SHARE(index_S);
	
	int8_t* ldg_A_SHARE = A_SHARE;
	int8_t* ldg_B_SHARE = B_SHARE;
	
	LDG1(data, weight, ldg_A_SHARE, ldg_B_SHARE, 0, batch_size); 
	data += 64 * 784;
	weight += 64 * 128;
	ldg_A_SHARE += (64 * 4 * 4 * 4 + 128 / 4 * 16);
	ldg_B_SHARE += (8192 + 256 / 4 * 16);
	LDG1(data, weight, ldg_A_SHARE, ldg_B_SHARE, 64, batch_size);
	__syncthreads();


	LDS_AB_1(A_SHARE, B_SHARE, A, B, index_S, 0, 0);
	A_SHARE +=  num_col * 16;
	B_SHARE +=  num_col * 32;

	//LDS_AB(A_SHARE, B_SHARE, A, B, index_S, 0, 0);
	LDS_AB_1(A_SHARE, B_SHARE, A, B, index_S, 0, 1);
	A_SHARE =  A_SHARE - num_col * 16 + num_col * 32;
	B_SHARE =  B_SHARE - num_col * 32 + num_col * 64;	



	MMA(&A[0], &B[0], Acc);
	MMA(&A[32], &B[32], Acc);
	__syncthreads();

	LDS_AB_1(A_SHARE, B_SHARE, A, B, index_S, 1, 0);
	A_SHARE +=  num_col * 16;
	B_SHARE +=  num_col * 32;		

	
	LDS_AB_1(A_SHARE, B_SHARE, A, B, index_S, 1, 1);

	MMA(&A[0], &B[0], Acc);
	MMA(&A[32], &B[32], Acc);


}

__device__ void QZ(int8_t* SC_SHARE, int8_t* A_SHARE, int* Acc, float* SCALE_SHARE, float* BIAS_SHARE, float scale_shortcut) {

	int tid = threadIdx.x;
	int lane = tid % 32;
	int warp = tid / 32;

	int row = lane / 4;  
	int col = lane % 4;  

	int8_t tmp_dst[2];
	float scale_r[2];
	float bias_r[2];
	int8_t tmp_shortcut[2];
	for (int j = 0; j < 4; j++) {  
		int index_col = col * 2 + warp / 2 * 32 + j * 8;
		reinterpret_cast<float2*>(&scale_r[0])[0] = reinterpret_cast<float2 const*>(&SCALE_SHARE[index_col])[0];
		reinterpret_cast<float2*>(&bias_r[0])[0] = reinterpret_cast<float2 const*>(&BIAS_SHARE[index_col])[0];
		for (int i = 0; i < 2; i++) {  
			for (int k = 0; k < 2; k++) {

				int index_share = (warp * 32 + i * 16 + k * 8 + row) * 32
					+ j * 8 + col * 2;
				reinterpret_cast<uint16_t*>(&tmp_shortcut[0])[0] = reinterpret_cast<uint16_t const*>(&SC_SHARE[index_share])[0];

				float shortcut = scale_shortcut * float(tmp_shortcut[0]);
				float tmp_result = shortcut + float(Acc[i * 4 + j * 8 + k * 2]) * scale_r[0] + bias_r[0];
				float tmp_max = tmp_result > 0.f ? tmp_result : 0.f;
				tmp_dst[0] = int8_t(tmp_max);
				shortcut = scale_shortcut * float(tmp_shortcut[1]);
				tmp_result = shortcut + float(Acc[i * 4 + j * 8 + k * 2 + 1]) * scale_r[1] + bias_r[1];
				tmp_max = tmp_result > 0.f ? tmp_result : 0.f;
				tmp_dst[1] = int8_t(tmp_max);

				reinterpret_cast<uint16_t*>(&A_SHARE[index_share])[0] = reinterpret_cast<uint16_t const*>(&tmp_dst[0])[0];
			}
		}
	}
}

__device__ void Res3Brc1(
	int8_t* data,
	int8_t* data_shortcut,
	int8_t* weight,
	int8_t* output,
	float* scale,
	float* bias,
	float scale_shortcut) {

	extern __shared__ int8_t SHARE[];

	int8_t* A_SHARE = &SHARE[0];  
	int8_t* B_SHARE = &A_SHARE[9216]; 
	int8_t* SC_SHARE = &B_SHARE[18432]; 
	float* SCALE_SHARE = reinterpret_cast<float*>(&SC_SHARE[8192]);
	float* BIAS_SHARE = &SCALE_SHARE[128];

	int Acc[32];
	for (int i = 0; i < 32; i++) {
		Acc[i] = 0;
	}

	LDGScale(SCALE_SHARE, scale, BIAS_SHARE, bias);
	LDGA(SC_SHARE, data_shortcut);
	
	//Pipline(data, weight, A_SHARE, B_SHARE, Acc);
	Pipline_ldmat(data, weight, A_SHARE, B_SHARE, Acc);
	
	__pipeline_wait_prior(0);
	__syncthreads();
	QZ(SC_SHARE, A_SHARE, Acc, SCALE_SHARE, BIAS_SHARE, scale_shortcut);
	__syncthreads();
	STG(output, A_SHARE);  //STG output

}


__device__ void Res3Brc2(
	int8_t* data,
	int8_t* data_shortcut,
	int8_t* weight,
	int8_t* output,
	float* scale,
	float* bias,
	float scale_shortcut,
	const int batch_size) {


	extern __shared__ int8_t SHARE[];

	int8_t* A_SHARE = &SHARE[0];  
	int8_t* B_SHARE = &A_SHARE[9216];  
	int8_t* SC_SHARE = &B_SHARE[18432];  
	float* SCALE_SHARE = reinterpret_cast<float*>(&SC_SHARE[8192]);
	float* BIAS_SHARE = &SCALE_SHARE[128];

	int Acc[32];
	for (int i = 0; i < 32; i++) {
		Acc[i] = 0;
	}

	LDGScale(SCALE_SHARE, scale, BIAS_SHARE, bias);
	LDGA1(SC_SHARE, data_shortcut, batch_size);
	Pipline1(data, weight, A_SHARE, B_SHARE, Acc, batch_size);
	__pipeline_wait_prior(0);
	__syncthreads();
	QZ(SC_SHARE, A_SHARE, Acc, SCALE_SHARE, BIAS_SHARE, scale_shortcut);

	__syncthreads();
	STG1(output, A_SHARE, batch_size);  //STG output
}


__global__ void Res3Brc(
	int8_t* data,
	int8_t* data_shortcut,
	int8_t* weight,
	int8_t* output,
	float* scale,
	float* bias,
	float scale_shortcut,
	const int batch_size) {
	int b_z = batch_size % 4;
	if (b_z == 0)
		Res3Brc1(data, data_shortcut, weight, output, scale, bias, scale_shortcut);
	else
		Res3Brc2(data, data_shortcut, weight, output, scale, bias, scale_shortcut, batch_size);
}


