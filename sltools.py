"""
saving the function using for save and load data

"""

import numpy as np
import pandas as pd
from  scipy.sparse import csc_matrix
import pickle




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


def save_pickle(matrix, filename):
    with open(filename, 'wb') as outfile:
        pickle.dump(matrix, outfile, pickle.HIGHEST_PROTOCOL)
def load_pickle(filename):
    with open(filename, 'rb') as infile:
        matrix = pickle.load(infile)
    return matrix



