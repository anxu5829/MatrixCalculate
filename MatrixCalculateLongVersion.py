import gc

gc.col_batchlect()

import pycuda
import pandas as pd

from pycuda import driver, compiler, gpuarray, tools
import numpy as np
import pycuda.autoinit



# it is better to let col_batch*attr < = 1e7

# 可以人为设定的是row_batch , col
row = 1024
batch_size = 1
row_batch =int( row / batch_size )
col = 1820099
col_batch_num = 5
col_batch =  int(col / col_batch_num) + 1
attr = 1
bandwith = 4
block = (row, 1, 1)
grid = (col_batch, batch_size, 1)

matrixCal_template = u"""
    #include <math.h>


    __global__ void matrixMulKernel(float *x , float *y  , float *z ){

              int   tx                =  threadIdx.x;
              int   bix               =  blockIdx.x ; 
        const long  COL_NUM           =  %(COL_NUM)s ;
        const int   COL_BATCH_NUM     =  %(COL_BATCH_NUM)s;
        const int   NUMOFATTR         =  %(NUMOFATTR)s;
        const long  col_batchOFOUT    =  gridDim.x ; //%(col_batchOFOUT)s;
        const int   BATCHNUM          =  blockDim.y;
        const int   ROWNUM            =  tx*BATCHNUM;
        //const long ROWOFOUT         =  blockDim.x ;   // %(ROWOFOUT)s;
        
        for(int i = 0 ; i != COL_BATCH_NUM ; i++){    
                bix = bix + col_batchOFOUT; 
                if(bix*NUMOFATTR+idx <= COL_NUM){
                    float s = 0;
                    for (int idx  = 0 ; idx< NUMOFATTR ; idx++){
                         s += pow(x[ROWNUM*NUMOFATTR+idx]-y[bix*NUMOFATTR+idx],2);
                        //s += pow(x[1]-y[1],2) ; 
                    }
                    z[ROWNUM*col_batchOFOUT+bix] =sqrt(s);
                }     
        }
    }



    __global__ void kernelCalculate(float *z){
              float bandwith          = %(bandwith)s ; 
              int   tx                =  threadIdx.x;
              int   bix               =  blockIdx.x ; 
        const long  COL_NUM           =  %(COL_NUM)s ;
        const int   COL_BATCH_NUM     =  %(COL_BATCH_NUM)s;
        const int   NUMOFATTR         =  %(NUMOFATTR)s;
        const long  col_batchOFOUT    =  gridDim.x ; //%(col_batchOFOUT)s;
        const int   BATCHNUM          =  blockDim.y;
        const int   ROWNUM            =  tx*BATCHNUM;
        //const long ROWOFOUT         =  blockDim.x ;   // %(ROWOFOUT)s;
        
        for(int i = 0 ; i != COL_BATCH_NUM ; i++){    
                bix = bix + col_batchOFOUT; 
                if(bix*NUMOFATTR+idx <= COL_NUM){
                    z[ROWNUM*col_batchOFOUT+bix]    =  z[ROWNUM*col_batchOFOUT+bix] / bandwith;
                }    
            }
    }

      __global__ void zsCal(float *z ,  float *s  , float *zs){

              int   tx                =  threadIdx.x;
              int   bix               =  blockIdx.x ; 
        const long  COL_NUM           =  %(COL_NUM)s ;
        const int   COL_BATCH_NUM     =  %(COL_BATCH_NUM)s;
        const int   NUMOFATTR         =  %(NUMOFATTR)s;
        const long  col_batchOFOUT    =  gridDim.x ; //%(col_batchOFOUT)s;
        const int   BATCHNUM          =  blockDim.y;
        const int   ROWNUM            =  tx*BATCHNUM;
        //const long ROWOFOUT         =  blockDim.x ;   // %(ROWOFOUT)s;
        
        for(int i = 0 ; i != COL_BATCH_NUM ; i++){    
                bix = bix + col_batchOFOUT; 
                if(bix*NUMOFATTR+idx < COL_NUM){
                    zs[ROWNUM*col_batchOFOUT+bix]   = s[ROWNUM*col_batchOFOUT+bix] * z[ROWNUM*col_batchOFOUT+bix] ; 
                }
            }

    }

"""

matrixCal = matrixCal_template % {
    'COL_BATCH_NUM':col_batch_num,
    'NUMOFATTR': attr,
    'col_batchOFOUT': col_batch,
    'ROWOFOUT': row,
    'bandwith': bandwith
}

# get function


mod = compiler.SourceModule(matrixCal)
matrixMul = mod.get_function("matrixMulKernel")
kernelCalculate = mod.get_function("kernelCalculate")
zsCal = mod.get_function("zsCal")

# model paras

# this x , y is used for test grid used


# prepare data here
# test for power and limit

# you must change the type to float32
x = np.random.random((row, attr)).astype(np.float32)
y = np.random.random((col, attr)).astype(np.float32)
r = np.random.random((col, 1)).astype(np.float32)
s = np.random.random((row, col)).astype(np.float32)

# test for algorithmn
# x             = np.arange(row*attr).reshape((row,attr)).astype(np.float32)
# y             = np.arange(10,col*attr+10).reshape((col,attr)).astype(np.float32)
# z             = np.empty((row,col))
# s             = np.arange(5,row*col+5).reshape((row,col)).astype(np.float32)
# r             = np.arange(col).reshape((col,1)).astype(np.float32)

# def main(x,y,r,s,row,col_batch,block,grid):

x_gpu = gpuarray.to_gpu(x)
y_gpu = gpuarray.to_gpu(y)
s_gpu = gpuarray.to_gpu(s)
z_gpu = gpuarray.empty((row, col), np.float32)
zs_gpu = gpuarray.empty((row, col), np.float32)

matrixMul(x_gpu, y_gpu, z_gpu, block=block, grid=grid)

kernelCalculate(z_gpu, block=block, grid=grid)

zsCal(z_gpu, s_gpu, zs_gpu, block=block, grid=grid)

zs = zs_gpu.get()

wr = zs.dot(r)
sum_zs = zs.dot(np.ones((col, 1)).astype(np.float32))

























