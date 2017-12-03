# prepare User distance square
import  numpy as np
import  pandas as pd
import  os
from    scipy.sparse import csr_matrix
from    scipy.sparse import csc_matrix
from    sklearn.preprocessing import LabelEncoder
import  sklearn.metrics.pairwise as smp
import  gc
from itertools import chain
import dask.array as da
import collections
import sys
from usefulTool import LargeSparseMatrixCosine

def subtractIdx(ObservationData , totalLen  ,encoded = True):
    """

    :param ObservationData: the data of (usr, item , value)
    :param totalLen is the number of observation
    :param encoded : using for determine  the usr , item had coded or not
    :return: usr list , item list , rlist , using in the process of calling distance
    """
    if encoded == True:
        # return a test tuple for next method
        ulist = np.arange(totalLen)
        ilist = np.arange(totalLen)
        rlist = np.arange(totalLen)
        return (ulist,ilist,rlist)

    else:
        pass

def kernelOperator(disSquare,bandwith =4 , ker  = "gauss"):
    """

    :param disSquare:  return the distance of Xui & X_omiga
    :param bandwith : the bandwith of kernel
    :param kernel :which kernel you may use
    :return:
    """
    if ker == "gauss":
        x = np.sqrt(disSquare) / bandwith
        core = np.exp(abs(x))
        return core

def yGenerator(ObjectNum,mode = "usr",parallel = True):
    global  usrDis, itemDis , usrNet , itemNet , ulist , ilist , rlist

    if parallel == True:
        pass
    else :
        if mode == "usr":
            uid = ObjectNum
            """
                 S_u_omiga : is the s for usr to omiga
                 S_i_omiga : is the s for item to omiga
                 d_u_omiga : is the dis of usr to omiga
                 d_i_omiga : is the dis of item to omiga
                                 
            """

            itemNum = itemDis.shape[0]
            S_u_omiga = np.array([ usrNet[ObjectNum,ulist] for _ in range(itemNum)])
            S_i_omiga =  itemNet[:,ilist]
            SocialNetWork = S_i_omiga * S_u_omiga

            d_u_omiga = np.array([ usrDis[ObjectNum,ulist] for _ in range(itemNum)])
            d_i_omiga = itemDis[:,ilist]

            disSquare = d_i_omiga * d_u_omiga

            kernel = kernelOperator(disSquare)

            y = kernel*SocialNetWork.dot(rlist)
            return(y)

        elif mode == "item":
            pass

############# method for data prepare#####################

# fill na
def fillNAN(tableName,values):
    # values = {'genre_ids' : 'unkown' }
    tableName = tableName.fillna(value = values)
    return  tableName


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


########################cal culate net #######################



def calculateDis(ObjectAttr, attrNum=2, ObjectNum=1e4):
    """

    :param ObjectAttr: must be a matrix has a shape like (ObjectNum *　attrNum)
    :param attrNum: the  number of continuous attr a usr have
    :param ObjectNum: the number of usr we observed
    :return:

    ObjectAttr matrix is ObjectNum * attrNum matrix
    target is to return a Object *　Object matrix  ， saving value of
    square  of distance  of each Object

    """
    # 　return a test matrix for other

    testx = np.arange(ObjectNum * attrNum).reshape((int(ObjectNum), attrNum))
    # 这里使用了itertools 往生成的矩阵添加东西
    # 思路是使用多进程完成任务
    dis = smp.pairwise_distances(testx)
    return dis ** 2


def calculateNet(ObjectAttr):
    """
    :param ObjectAttr: must be a matrix has a shape like (ObjectNum *　attrNum)
    :param attrNum: the  number of categorical attr a usr have
    :param ObjectNum: the number of usr we observed
    :return:

    ObjectAttr matrix is ObjectNum * attrNum matrix
    target is to return a Object *　Object matrix  ， saving value of
    the social net work of each Object
    """
    # 　return a test matrix for other
    attrNum = ObjectAttr.shape[1]
    ObjectNum = ObjectAttr.shape[0]



    #
    # itemAttrNum_da = da.from_array(itemAttrNum, chunks=(1000, 1000))
    #
    # # calculate the dot
    # tagNum = itemTagmatrix.shape[1]
    # itemNum = itemTagmatrix.shape[0]
    # item_item_matrix = csc_matrix((itemNum, itemNum))
    # item_item_matrix = da.from_array(item_item_matrix, chunks=(1000, 1000))
    # dotBatch = np.arange(0, tagNum, 100)
    # for i in range(len(dotBatch)):
    #     if i != (len(dotBatch) - 1):
    #         itemTagmatrix_da = da.from_array(itemTagmatrix[:, dotBatch[i]:dotBatch[i + 1]], chunks=(1000, 1000))
    #     else:
    #         itemTagmatrix_da = da.from_array(itemTagmatrix[:, dotBatch[i]:], chunks=(1000, 1000))
    #     item_item_matrix += itemTagmatrix_da.dot(itemTagmatrix_da.transpose())
    #
    # gc.collect()
    #
    # item_item_matrix = item_item_matrix / itemAttrNum_da
    # item_item_matrix = item_item_matrix.transpose() / itemAttrNum_da
    # item_item_matrix = item_item_matrix.transpose()
    #
    # da.to_hdf5('D:\\item.hdf5', '/item_itemNet/data', item_item_matrix)

    pass


###############################data prepare ################





if __name__=="__main__":


    # change dir
    os.chdir("C:\\Users\\22560\\Desktop\\recommand Sys\\recommand Sys")

    # load item data
    item = pd.read_csv("songsCSV.csv",encoding="UTF-8" ,dtype = {
        "song_length": np.uint16,
        "language" : str
    })

# use for debug
    item.loc[2296320,'song_id'] = 'special'

    # fill na use default value , this value is also used in build social network
    # be caution ! you just need to fill those cols will be used in
    # building the network!
    fillnawith = collections.OrderedDict()
    fillnawith['genre_ids'] = '-1'
    fillnawith['language'] = '-1'
    fillnawith['artist_name'] = "no_artist"


    item = fillNAN(item, fillnawith)

    # fill na with special value calculated from data
    pass


    # change primary key to ID
    item , song_id = changeNameToID(item, 'song_id' , plan = "A")


    # split the dataframe to two , one of it is containing  the continue attr
    # the other containing the discrete attr

    # 注意：分类不能分的太细，分之前可以对于属性做做聚类，把歌手这种东西先聚聚类，否则分类太多
    #　一是超大矩阵运算不好做，二来分的太细做社交网络就没意义了


    ( itemCntnueAttr , itemDscrtAttr ) = \
        splitDF(item,"song_id",
                ["song_length"],
                ["genre_ids","language","artist_name"]
                )
    del item ; gc.collect();



    # create socail network of item using dask

    # do the tag combine process

    id = "song_id"
    colList = itemDscrtAttr.columns.tolist()
    colList.remove(id)

    itemWithTag = tagCombine(itemDscrtAttr, id='song_id', tagColList=colList)

    (itemTagmatrix,itemNoAttr) = findNetwork(itemWithTag,  fillnawith , split = r"&|\|")

    LargeSparseMatrixCosine(itemTagmatrix,num=5000)



    # usrNet = calculateNet(1,1,1*1e4)



    # generate two dis for test later method
    usrDis = calculateDis(1,2,1*1e4)
    itemDis = calculateDis(1,2,22*1e4)



    (ulist, ilist , rlist) = subtractIdx(1,180*1e4,encoded=True)

