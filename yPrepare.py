# prepare User distance square
import  numpy as np
import  pandas as pd
import  os

import  gc

import collections


from usefulTool import LargeSparseMatrixCosine
from usefulTool import mergeDataToSparse
from usefulTool import saveToH5
from usefulTool import fillDscrtNAN
from usefulTool import fillCntnueNAN
from usefulTool import changeNameToID
from usefulTool import splitDF
from usefulTool import tagCombine
from usefulTool import findNetwork



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


    item = fillDscrtNAN(item, fillnawith)


    # fill na with special value calculated from data

    fillCntnueNAN(item, ['song_length'])


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


# use for test
    itemTagmatrix = itemTagmatrix[:100000,:]


    # if you want to do it using loop , you may set num > 2
    # if you set num = 2 ,it will do it once
    LargeSparseMatrixCosine(itemTagmatrix,num=10,fileplace="C:\\Users\\22560\\Desktop\\")



    itemNet = mergeDataToSparse(workfilename="C:\\Users\\22560\\Desktop\\",numOfFile = 1)



    saveToH5(itemNet)





    # usrNet = calculateNet(1,1,1*1e4)
    #
    #
    #
    # # generate two dis for test later method
    # usrDis = calculateDis(1,2,1*1e4)
    # itemDis = calculateDis(1,2,22*1e4)
    #
    #
    #
    # (ulist, ilist , rlist) = subtractIdx(1,180*1e4,encoded=True)
    #
    #
