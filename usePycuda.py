import  pycuda
from pycuda import driver , compiler , gpuarray , tools
import pandas as pd
import numpy as np




import pycuda.autoinit

x = np.array(
    [
        [1,2,3,4,0],
        [2,3,4,5,0]

    ]

).astype(np.float32)

y =  np.array(
    [
        [1,2,3,4,0],
        [2,3,4,5,0],
        [3,4,2,4,1],
        [0,2,3,4,1]
    ]

).astype(np.float32)

z = np.empty((2,4))

x_gpu = gpuarray.to_gpu(x)

y_gpu = gpuarray.to_gpu(y)

z_gpu = gpuarray.empty((2,4),np.float32)






kernel_node_template = """



__glabal__ void MatrixMulKernel(float *x , float *y  , float *z ){
    /*
    int tx  = threadIdx.x;
    int ty  = threadIdx.y;
    int tz  = 5;
    int s = 0;
    for (int idx  = 0 ; idx<tz ; idx++){
         s += (x[tx*tz+idx]-y[ty*tz+idx])^2;
    }
    
    z[tx*4+ty] = s ; 
    */
}

"""
mod = compiler.SourceModule(kernel_node_template)

matrixMul = mod.get