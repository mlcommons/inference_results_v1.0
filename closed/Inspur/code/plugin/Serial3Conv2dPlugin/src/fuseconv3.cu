//#pragma once
#include <iostream>
#include "LDG.h"
#include "Res3brc.h"


#include <mma.h>

using namespace std;



void fuseconv1(
                int I_N,  //batch
                int8_t* data,
                int8_t* shortcut_data,
                int8_t* weight,
                int8_t* output,
                float* scale,
                float* bias,
                float scale_shortcut,
                cudaStream_t stream) {


    dim3 block;
    dim3 grid;
    block.x=256;
    grid.x = 4;
    grid.y = 49;
    grid.z = (I_N+4-1)/4;

    int num_share=36*1024;
    int smem_size = int(num_share * sizeof(int8_t));
    cudaFuncSetAttribute(
                Res3Brc,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_size);

        Res3Brc<<<grid, block, num_share/4*sizeof(int),stream>>>(data,shortcut_data,weight,output,scale,bias,scale_shortcut,I_N);
    

}

