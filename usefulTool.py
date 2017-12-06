import numpy as np
import pandas as pd
from  scipy.sparse import csc_matrix
from  scipy.sparse import diags
import gc
import h5sparse


def sparseToPandas(sparseMatrix):
    coo = sparseMatrix.tocoo(copy = False)
    data = pd.DataFrame(
        {'row': coo.row,
         'col': coo.col,
         'data': coo.data}
        )[['row', 'col', 'data']].\
        sort_values(['row', 'col']).\
        reset_index(drop=True)
    return(data)


def pandasToSparse(pandasMatrix):
    csc = csc_matrix((pandasMatrix.data,
                     (pandasMatrix.row,pandasMatrix.col)))
    return csc




# save sparse matrix using numpu method
def save_sparse_csc(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

# load sparse matrix
def load_sparse_csc(filename):
    loader = np.load(filename)
    return csc_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])




# save / load data using cPickle
import pickle

def save_pickle(matrix, filename):
    with open(filename, 'wb') as outfile:
        pickle.dump(matrix, outfile, pickle.HIGHEST_PROTOCOL)
def load_pickle(filename):
    with open(filename, 'rb') as infile:
        matrix = pickle.load(infile)
    return matrix




def LargeSparseMatrixCosine(largeSparseMatrix,num = 5000,select = 0.7,fileplace = "D:\\tempdata\\"):

    # this method will save the result in disk
    (rowNum,colNum) = largeSparseMatrix.shape
    sep = np.linspace(0,rowNum,endpoint=True,dtype=np.int64,num=num)
    # calculate the L2 of each vector
    lenOfVec = np.sqrt(largeSparseMatrix.multiply(largeSparseMatrix).\
                            sum(axis=1).A.ravel()
                            )
    lenOfVecAll = diags(1 / lenOfVec)
    for i,j in enumerate(sep):
        if i+1 < len(sep):
        #if i < 40:
            #print(i,j)
            # get a block from the ogininal matrix
            block_of_sparse = largeSparseMatrix[j:sep[i+1],:]

            # calculate the dot
            dot_product_of_block = block_of_sparse.dot(
                                        largeSparseMatrix.transpose()
                                    )
            lenOfBlockVec = diags(1/ lenOfVec[j:sep[i+1]])

            dot_cosine = lenOfBlockVec @ dot_product_of_block @ lenOfVecAll

            #　we just select few of the to build net work
            dot_cosine = dot_cosine > select

            gc.collect()

            dot_cosine = sparseToPandas(dot_cosine)
            dot_cosine.row = dot_cosine.row + j

            dot_cosine.to_csv(
                    fileplace + "dot_cosine"+str(i)+".gzip",sep = ',',
                    index = False,
                    encoding= "utf-8",
                    compression = "gzip"
                              )

            del dot_cosine,lenOfBlockVec
            gc.collect()
            print("S_item is now  preparing")
            print( str((i+1)/len(sep)) +"percent of data is prepared ")







def largeMatrixDis(ObjectDis,id ):
    # id = "song_id"
    pass







def mergeDataToSparse(workfilename,numOfFile):
    # workfilename = "D:\\tempdata\\"
    list = []
    for i in range(numOfFile):
        file = pd.read_csv(workfilename+"dot_cosine"+str(i) +".gzip" ,compression= "gzip")
        list.append(file)
        del file

    dot_cosine = pd.concat(list)
    dot_cosine = pandasToSparse(dot_cosine)
    return dot_cosine





# api :https://pypi.python.org/pypi/h5sparse/0.0.4
def saveToH5(sparseMatrix,filename):
    with h5sparse.File(filename) as h5f:
        h5f.create_dataset('sparse/matrix', data=sparseMatrix)



def laodFromH5(filename,filedir):
    h5f = h5sparse.File(filename)
    return h5f[filedir]


    # h5f['sparse/matrix'][1:3].toarray()
    # h5f['sparse']['matrix'][1:3].toarray()
    # h5f['sparse']['matrix'].value.toarray()
    # this one is allow you to append data ， nice ！

