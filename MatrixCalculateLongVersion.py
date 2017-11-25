import gc

gc.collect()

import pycuda
import pandas as pd

from pycuda import driver, compiler, gpuarray, tools
import numpy as np
import pycuda.autoinit



# it is better to let col_batch*attr < = 1e7

# 可以人为设定的是row_batch , col
row = 100
batch_size = 1
row_batch =int( row / batch_size )
col = 1800000
col_batch_num = 100
col_batch =  int(col / col_batch_num) + 1
attr = 2
bandwith = 4
block = (row, 1, 1)
grid = (col_batch, batch_size, 1)




# model paras

# this x , y is used for test grid used


# prepare data here
###### test for power and limit

# you must change the type to float32
# x = np.random.random((row, attr)).astype(np.float32)
# y = np.random.random((col, attr)).astype(np.float32)
# r = np.random.random((col, 1)).astype(np.float32)
# s = np.random.random((row, col)).astype(np.float32)

##### test for algorithmn
x        = np.arange(row*attr).reshape((row,attr)).astype(np.float32)
y        = np.arange(10,col*attr+10).reshape((col,attr)).astype(np.float32)
z        = np.empty((row,col)).astype(np.float32)
s        = np.arange(5,row*col+5).reshape((row,col)).astype(np.float32)
r        = np.arange(col).reshape((col,1)).astype(np.float32)

# def main(x,y,r,s,row,col_batch,block,grid):

x_gpu = gpuarray.to_gpu(x)
y_gpu = gpuarray.to_gpu(y)
s_gpu = gpuarray.to_gpu(s)
z_gpu = gpuarray.empty((row, col), np.float32)
zs_gpu = gpuarray.empty((row, col), np.float32)




matrixCal_template = u"""
    #include <math.h>


    __global__ void matrixMulKernel(float *x , float *y  , float *z ){
        /*
        tx : 对应的是threadIdx.x 
        bix : 正在处理的列号
        ROWNUM : 对应 小batch 处理的行号,对应到z的行号
        COL_NUM : 表示总的处理的列的数量
        COL_BATCH_NUM : 对应总共处理几轮
        NUMOFATTR : 对应x的属性的个数
        col_batchOFOUT : 对应一个batch的处理的列的数量
        BATCHNUM : 对应处理的行的轮数
        ROWOFOUT : 一个block 处理的thread 的个数 
        
        */
              int   tx                =  threadIdx.x;
              int   bix               =  blockIdx.x ; 
        const long  COL_NUM           =  %(COL_NUM)s ;
        const int   COL_BATCH_NUM     =  %(COL_BATCH_NUM)s;
        const int   NUMOFATTR         =  %(NUMOFATTR)s;
        const long  col_batchOFOUT    =  gridDim.x ; //%(col_batchOFOUT)s;
        const int   BATCHNUM          =  blockDim.y;
        const int   ROWNUM            =  tx*BATCHNUM;
        //const long  ROWOFOUT           =  blockDim.x ;   // %(ROWOFOUT)s;
        
        
        
        bix = bix - col_batchOFOUT; 
        for(int i = 0 ; i != COL_BATCH_NUM ; i++){    
                bix = bix + col_batchOFOUT; 
                float s = 0;
                for (int idx  = 0 ; idx< NUMOFATTR ; idx++){
                        if(bix <= COL_NUM){
                             s += pow(x[ROWNUM*NUMOFATTR+idx]-y[bix*NUMOFATTR+idx],2);
                             //s += pow(x[1]-y[1],2) ; 
                        }

                }  
                if(bix < COL_NUM){
                      z[ROWNUM*COL_NUM+bix] = sqrt(s);
                }   
        }
        
    }



    __global__ void kernelCalculate(float *z){
              float bandwith          = %(bandwith)s ; 
              int   tx                =  threadIdx.x;
              int   bix               =  blockIdx.x ; 
        const long  COL_NUM           =  %(COL_NUM)s ;
        const int   COL_BATCH_NUM     =  %(COL_BATCH_NUM)s;
        //const int   NUMOFATTR         =  %(NUMOFATTR)s;
        const long  col_batchOFOUT    =  gridDim.x ; //%(col_batchOFOUT)s;
        const int   BATCHNUM          =  blockDim.y;
        const int   ROWNUM            =  tx*BATCHNUM;
        //const long  ROWOFOUT          =  blockDim.x ;   // %(ROWOFOUT)s;
        bix = bix - col_batchOFOUT; 
        for(int i = 0 ; i != COL_BATCH_NUM ; i++){    
                bix = bix + col_batchOFOUT; 
                if(bix < COL_NUM){
                    z[ROWNUM*COL_NUM+bix]  =  z[ROWNUM*COL_NUM+bix] / bandwith;
                }  
            }
    }

      __global__ void zsCal(float *z ,  float *s  , float *zs){

              int   tx                =  threadIdx.x;
              int   bix               =  blockIdx.x ; 
        const long  COL_NUM           =  %(COL_NUM)s ;
        const int   COL_BATCH_NUM     =  %(COL_BATCH_NUM)s;
        //const int   NUMOFATTR         =  %(NUMOFATTR)s;
        const long  col_batchOFOUT    =  gridDim.x ; //%(col_batchOFOUT)s;
        const int   BATCHNUM          =  blockDim.y;
        const int   ROWNUM            =  tx*BATCHNUM;
        //const long  ROWOFOUT          =  blockDim.x ;   // %(ROWOFOUT)s;
        bix = bix - col_batchOFOUT; 
        for(int i = 0 ; i != COL_BATCH_NUM ; i++){    
                bix = bix + col_batchOFOUT; 
                if(bix <= COL_NUM){
                    zs[ROWNUM*COL_NUM+bix]   = s[ROWNUM*COL_NUM+bix] * z[ROWNUM*COL_NUM+bix] ; 
                }
            }

    }

"""

matrixCal = matrixCal_template % {
    'COL_NUM':col,
    'COL_BATCH_NUM':col_batch_num,
    'NUMOFATTR': attr,
    'col_batchOFOUT': col_batch,
    'ROWOFOUT': row,
    'bandwith': bandwith
}

# get function


model = compiler.SourceModule(matrixCal)

matrixMulSuper = model.get_function("matrixMulKernel")

kernelCalculateSuper = model.get_function("kernelCalculate")

zsCalSuper = model.get_function("zsCal")




matrixMulSuper(x_gpu, y_gpu, z_gpu, block=block, grid=grid)

kernelCalculateSuper(z_gpu, block=block, grid=grid)

zsCalSuper(z_gpu, s_gpu, zs_gpu, block=block, grid=grid)


zs = zs_gpu.get()
zs = zs.transpose() / zs.sum(1)
zs = zs.transpose()



wr = zs.dot(r)
























