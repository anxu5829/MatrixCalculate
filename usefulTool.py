import pickle
import h5sparse
import gc
import sys
import dask.array as da
import numpy as np
import pandas as pd
from    scipy.sparse          import csc_matrix
from    scipy.sparse          import diags
from    scipy.sparse          import csr_matrix
from    scipy.sparse          import csc_matrix
from    sklearn.preprocessing import LabelEncoder
from    sltools               import pandasToSparse
from    sltools               import sparseToPandas



"""
data preparaton

"""

# fill na
def fillDscrtNAN(tableName,values):
    # values = {'genre_ids' : 'unkown' }
    tableName = tableName.fillna(value = values)
    return  tableName






def fillCntnueNAN(tableName,values):
    # a list of continue col who will be fill with mean
    # values = ['song_length']
    tableName.loc[:, values] = tableName[values].fillna(tableName[values].mean())





def changeNameToID(tableName,id , plan = 'A'):
    """

    :param tableName: the table name
    :param id: the primary id
    :param plan:plan A uses category.cat.codes to change the id , plan B use sklearn's encoder to encode id
    :return: return a tuple ,of which first is the table using encoded id , second is the original id
    """
    if plan == 'A':

        originalName = tableName[id]

        originalName_index = originalName.index

        originalNameUnique = originalName.unique()

        originalNameCategory = originalName.astype("category", categories=originalNameUnique).cat.codes

        tableName[id] = originalNameCategory

        del originalName_index,originalNameUnique,originalNameCategory

        return(tableName,originalName)

    elif plan == 'B':

        le = LabelEncoder()

        le.fit(tableName[id])

        originalName = tableName[id]

        tableName[id] = le.transform(tableName[id])

        return (tableName,originalName)


def splitDF(tableName,id,colListContinue,colListDiscrete):
    colListContinue.append(id)
    colListDiscrete.append(id)
    itemCntnueAttr  = tableName[colListContinue].copy()
    itemDscrtAttr   = tableName[colListDiscrete].copy()
    return (itemCntnueAttr,itemDscrtAttr)


def tagCombine(tableName, tagColList, id, split="|"):
    #temp  =   tableName.set_index(id)
    for i in tagColList:
        tableName.loc[:,i] = tableName[i].str.strip()

    tag0 = tagColList[0]
    temp = pd.DataFrame()
    temp["concat"] = tableName[tag0].str.cat([ tableName[i] for i in tagColList if i != tag0],sep = split)
    temp = temp.set_index(tableName[id])
    return(temp)



def findNetwork(tableName,fillnawith,split = r"&|\|" ):

    #tableName = songtag;split = "|" ,

    # select useful columns
    temp = tableName.copy()
    temp.reset_index(inplace =True)
    idname = tableName.index.name
    id = temp[idname]


    # split colName based on var split , due to orignal data has sth like "a|b"
    temp['splitTag'] = temp['concat'].str.split(split)

    contains = r''+'.*' + '.*'.join(fillnawith.values()) + '.*' + ''

    temp['hasNoTag'] = temp.concat.str.contains(contains)

    temp.drop( ["concat"] ,axis = 1 , inplace=True)


    temp['lenOfType'] = temp['splitTag'].map(lambda  x : len(x))

    objectHasNoTag = (id[temp.hasNoTag == True])


    colNameSpread = [ j  for i in temp['splitTag'] for j in i]
    idSpread = [ [id[i]]*j for i,j in enumerate(temp['lenOfType'])]
    idSpread = [j for i in idSpread for j in i]

    # id_colNameDF has the rows which contain the id-tag pair
    id_colNameDF = pd.DataFrame({idname:idSpread,'tag':colNameSpread})

    del temp,idSpread,colNameSpread
    gc.collect()


    # use label encoder to transform genre_id so that we can build a spasre - matrix

    # id_colNameDF[id_colNameDF[id].isin(objectHasNoTag)]

    id_colNameDF = id_colNameDF[-id_colNameDF.tag.isin([ str(i)  for  i in fillnawith.values()])]


    id_colNameDF ,colNameList = changeNameToID(id_colNameDF,'tag' ,plan="B" )


    id_colNameDF[idname] = id_colNameDF[idname].astype(int)
    id_colNameDF['tag'] = id_colNameDF['tag'].astype(int)

    #flag = id_colNameDF.ix[colNameList.astype(int) == -1,'genre_ids'].values[0]


    #id_colNameDF.ix[colNameList.astype(int) == -1,'target'] = -1
    #id_colNameDF.ix[colNameList in fillnawith.values() , 'target'] = -1

    maxcoder = max(id_colNameDF.tag.unique())
    id_colNameDF2 = pd.DataFrame({idname:objectHasNoTag,'tag':maxcoder+1})

    id_colNameDF['target'] = 1
    id_colNameDF2['target'] = -1

    id_colNameDF = pd.concat([id_colNameDF,id_colNameDF2])

    del id_colNameDF2
    gc.collect()
    ObjectTagmatrix = csc_matrix((id_colNameDF['target'], (id_colNameDF[idname], id_colNameDF['tag'])))
    del id_colNameDF
    gc.collect()

    return(ObjectTagmatrix,objectHasNoTag)







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

