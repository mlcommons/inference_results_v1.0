#pragma once

#include "fuseconv3.h"
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <mma.h>
//#include "Res3Brc.h"


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
                cudaStream_t stream) ;

