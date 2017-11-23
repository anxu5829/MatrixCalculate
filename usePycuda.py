import  pycuda
from pycuda import driver , compiler , gpuarray , tools
import pandas as pd
import numpy as np




import pycuda.autoinit

# this x ,  y is used for test function
# x = np.array(
#     [
#         [1,2,3,4,0],
#         [2,3,4,5,0]
#
#     ]
#
# ).astype(np.float32)
#
# y =  np.array(
#     [
#         [1,2,3,4,0],
#         [2,3,4,5,0],
#         [3,4,2,4,1],
#         [0,2,3,4,1]
#     ]
#
# ).astype(np.float32)

# z = np.empty((2,4))


# this x , y is used for test grid used
row = 100
col = 100
x = np.random.random((row,5)).astype(np.float32)
y = np.random.random((col,5)).astype(np.float32)
z = np.empty((row,col))
x_gpu = gpuarray.to_gpu(x)

y_gpu = gpuarray.to_gpu(y)

z_gpu = gpuarray.empty((2,4),np.float32)



kernel_node_template = u"""
    #include <math.h>

    __global__ void matrixMulKernel(float *x , float *y  , float *z ){
      

    int tx  = threadIdx.x;
    int ty  = threadIdx.y;
    int tz  = 5;
    float s = 0;
    for (int idx  = 0 ; idx< tz ; idx++){
         s += pow(x[tx*5+idx]-y[ty*5+idx],2);
    }
    
    z[tx*4+ty] = sqrt(s) ; 
    
    
    int bix = blockIdx.x ; 
    int bdx = blockDim.x ;
    z[0] =  bix ;
    z[1] =  bdx ; 
    
}

"""
mod = compiler.SourceModule(kernel_node_template)

matrixMul = mod.get_function("matrixMulKernel")


matrixMul(x_gpu,y_gpu,z_gpu,block = (row,col,1), grid = (1,1))








#
# import pycuda.autoinit
# import pycuda.driver as drv
# import numpy
#
# from pycuda.compiler import SourceModule
#
# mod = SourceModule("""
# __global__ void multiply_them(float *dest, float *a, float *b)
# {
#   const int i = threadIdx.x;
#   dest[i] =   blockIdx.x;
#
# }
# """)
#
# multiply_them = mod.get_function("multiply_them")
#
# a = numpy.random.randn(400).astype(numpy.float32)
# b = numpy.random.randn(400).astype(numpy.float32)
#
# dest = numpy.zeros_like(a)
# multiply_them(
#     drv.Out(dest), drv.In(a), drv.In(b),
#     block=(100, 2, 1), grid=(2, 2))
#
# print(dest)


#
# __global__ void add( int *a, int *b, int *c ) {
#     int tid = threadIdx.x + blockIdx.x * blockDim.x;
#     while (tid < N) {
#         c[tid] = a[tid] + b[tid];
#         tid += blockDim.x * gridDim.x;
#     }
# }
