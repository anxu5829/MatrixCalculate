# use dask to cal and save data


import  numpy as np

import dask.array as da

import os


os.chdir("D:")
#os.chdir("C:\\Users\\22560\\Desktop\\recommand Sys\\recommand Sys")


# generate test data
z=np.arange(2e5)
z = z.reshape((int(1e5),2))

# this will rise an  memory error
#z = z.dot(z.transpose())

#we can do it like this

#change to da
z = da.from_array(z,chunks=(10000,2))

# use da dot method
#　秒算
zdot = z.dot(z.transpose())









# 可以很方便的从zdot中提取数据
arrayNeed = zdot[1,:].compute()


# 存储和调用
# save data so that can be use repeatly
# 真实数据大概是存不下的，可能需要>100g的存储空间，
# 电脑水平不够

da.to_hdf5('data.hdf5','/zdata/zdot',zdot)



# load data from hdf5
import h5py
f = h5py.File("data.hdf5")
# 似乎可以只读取局部数据？
# d is a pointer to the data on disk
d = f['/zdata/zdot']
# you can extract data in d like this
data = d[1:20,:]


# test big multiply

dd = da.from_array(d,chunks=(10000,10000))

x = np.ones((2,100000))
y = x.dot(dd)



# 关于hdf5 的管理
# remember : Groups work like dictionaries,
# and datasets work like NumPy arrays

import h5py
import numpy as np
f = h5py.File("mytestfile.hdf5", "w")

# create an empty 'array' on your disk
# (100,) is its shape
# "mydatasets" is the name of this 'array'
# dset is like a pointer so that you can manipulate it
dset = f.create_dataset("mydataset", (100,), dtype='i')

dset[:] = np.arange(100)

# it has some attr
dset.shape
dset.dtype

# you can use it like array

dset[:10]

# how hdf5 organize data?
# it organize data using their 'name' ,
# like you organize your file on your pc
dset.name

# you can add some meta data on dset:
dset.attrs['create Time'] = '2017-11-26'


# you can obtain part of data from hdf5
# this will create a pointer
pointer = f['mydataset']
# slice method will load data into mem
pointer[:10]


# and 'f' also has a name , point to the root
f.name


# you can use create_group to create a "folder":

xuan = f.create_group("xuan")

# now var xuan is point to the 'folder' '/xuan'
z = np.arange(10)
z_hdf5 = xuan.create_dataset("z",data = z  )

# if you want to get the content under the 'folder':
[i for  i in f.items()]

# if you want to drop a folder/dataset :

del f['xuan']


# 关于odo
import blaze as bz
import pandas as pd
x= bz.data("test.csv")
x = bz.odo(x,pd.DataFrame)



# 关于dask：一个强大的数据处理模块

# 1 array 方法
## create and store

# generate test data
z=np.arange(2e4)
z = z.reshape((int(1e4),2))

# this will rise an  memory error
#z = z.dot(z.transpose())

#we can do it like this

#change to da
z = da.from_array(z,chunks=(1000,2))


# store

z.to_hdf5("z.hdf5","/z",z)

# about sparse matrix:
import dask.array as da
import numpy as np
import sparse
from scipy.sparse import csr_matrix

sprs = csr_matrix((1e4, 1e4))
sprs[1,1] = 1000
sprs_da = da.from_array(sprs,chunks=(10,2))

# 注意：map_blocks 本质是对于每一个chunk 做一个函数映射，理论上可以自定义各种复杂函数
sprs_da = sprs_da.map_blocks(sparse.COO)

# 2 dataframe 方法
#A Dask DataFrame is a large parallel dataframe
# composed of many smaller Pandas dataframes,
# split along the index.

# create and store
import dask.dataframe as dd

# dd 不会直接把数据读到内存中
train = dd.read_csv("train.csv")

train.head()

# dd 不仅有dataframe , 还有series

# api
# you need to learn it someday
# http://dask.pydata.org/en/latest/dataframe-api.html#series





# 3 delayed 方法：并行运算

# 实现了计算图
from dask import delayed
def inc(x):
    return x + 1

def double(x):
    return x + 2

def add(x, y):
    return x + y

data = [1, 2, 3, 4, 5]


output = []
for x in data:
    a = delayed(inc)(x)
    b = delayed(double)(x)
    c = delayed(add)(a, b)
    output.append(c)

total = delayed(sum)(output)
total.compute()


# 可以设计图：

# 等待上一步的运算结果
import numpy as np

def cal1(a,b):
    return a+b
def cal2(c,d):
    return c*d
a = np.array([1,2])
b = np.array([3,4])
d = 10



c = delayed(cal1)(a,b)
e = delayed(cal2)(c,d)
f = delayed(sum)(e)
f.visualize()

#delayed

# future : 可以进行并行计算
from dask.distributed import Client


# 使其满足多线程运算





# 4 bag 方法：用于处理一大群零碎的文件（暂略）











# dask 的一个应用

from sklearn import linear_model
from dask import delayed
from dask import compute
import numpy as np
import dask

reg = linear_model.LinearRegression()

Y = np.random.random((50,4))

X = np.random.random((50,3))

result = []

def regression(X,Y):
    t = reg.fit(X,Y)
    return(t.coef_)


def concat(result):
    return result

for i in range(4):
    c = delayed(regression)(X,Y[:,i])
    result.append(c)

r = delayed(concat)(result)

r.compute()
r.visualize()




# regression

import dask.array as da

y = d[10,:]

xt = np.arange(20e4).reshape((2,int(10e4)))









