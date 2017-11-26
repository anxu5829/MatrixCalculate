# use dask to cal and save data


import  numpy as np
import dask.array as da
import os


os.chdir("C:\\Users\\22560\\Desktop\\recommand Sys\\recommand Sys")


# generate test data
z=np.arange(2e4)
z = z.reshape((int(1e4),2))

# this will rise an  memory error
#z = z.dot(z.transpose())

#we can do it like this

#change to da
z = da.from_array(z,chunks=(1000,2))

# use da dot method
#　秒算
zdot = z.dot(z.transpose())


# 可以很方便的从zdot中提取数据
arrayNeed = zdot[1,:].compute()


# 存储和调用
# save data so that can be use repeatly
# 真实数据大概是存不下的，可能需要>100g的存储空间，
# 电脑水平不够
da.to_hdf5('data.hdf5','/zdata/z2',z)
da.to_hdf5('data.hdf5','/zdata/zdot2',zdot)


# load data from hdf5
import h5py
f = h5py.File("data.hdf5")
# 似乎可以只读取局部数据？
# d is a pointer to the data on disk
d = f['/zdata/zdot2']
# you can extract data in d like this
data = d[1:]



# 关于hdf5 的管理
