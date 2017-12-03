import numpy as np
import pandas as pd
from  scipy.sparse import csc_matrix
from  scipy.sparse import diags
import gc


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




def LargeSparseMatrixCosine(largeSparseMatrix,num = 3000):
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
            #print(i,j)
            # get a block from the ogininal matrix
            block_of_sparse = largeSparseMatrix[j:sep[i+1],:]

            # calculate the dot
            dot_product_of_block = block_of_sparse.dot(
                                        largeSparseMatrix.transpose()
                                    )
            lenOfBlockVec = diags(1/ lenOfVec[:sep[i+1]])

            dot_cosine = lenOfBlockVec @ dot_product_of_block @ lenOfVecAll


            filename = "dot_cosine_"+str(j)+"_"+str(sep[i+1])+".mtx"
            save_sparse_csc(
                        filename,dot_cosine
                        )

            del dot_cosine,lenOfBlockVec
            gc.collect()



