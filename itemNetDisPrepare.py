import  os
import  gc
import collections
import h5py
import h5sparse
# prepare User distance square
import  numpy as np
import pandas as pd
from usefulTool import LargeSparseMatrixCosine
from usefulTool import fillDscrtNAN
from usefulTool import fillCntnueNAN
from usefulTool import scaleCntnueVariable
from usefulTool import changeNameToID
from usefulTool import splitDF
from usefulTool import tagCombine
from usefulTool import findNetwork
from usefulTool import largeMatrixDis
from usefulTool import DealingDescreteMissingValue
from sltools    import save_pickle





def extractItemInfo():
    ####################           change dir       ######################
    os.chdir("C:\\Users\\22560\\Desktop\\recommand Sys\\recommand Sys")

    # load item data
    item = pd.read_csv("songsCSV.csv", encoding="UTF-8", dtype={
        "song_length": np.uint16,
        "language": str,

    },iterator= True)

    # use for debug
    chunksize = 2000 #100000
    item = item.get_chunk(chunksize)
    item.loc[chunksize+1, 'song_id'] = 'specialll'
    item.loc[chunksize+2, 'song_id'] = 'special22'


    # change primary key to ID
    item, item_id_dict = changeNameToID(item, 'song_id', plan="A")



    # fill na use default value , this value is also used in build social network
    # be caution ! you just need to fill those cols will be used in
    # building the network!
    # and you must fillna with the order which is determined by the order of
    # those cols in your data
    fillnawith = collections.OrderedDict()
    fillnawith['genre_ids'] = '-1'
    fillnawith['language'] = '-1'
    fillnawith['artist_name'] = "no_artist"

    item = fillDscrtNAN(item, fillnawith)

    # fill na with special value calculated from data

    itemHasntCntnue = fillCntnueNAN(item, ['song_length'],'song_id')
    scaleCntnueVariable(item,['song_length'])



    # split the dataframe to two , one of it is containing  the continue attr
    # the other containing the discrete attr

    # 注意：分类不能分的太细，分之前可以对于属性做做聚类，把歌手这种东西先聚聚类，否则分类太多
    # 　一是超大矩阵运算不好做，二来分的太细做社交网络就没意义了


    (itemCntnueAttr, itemDscrtAttr) = \
        splitDF(item, "song_id",
                ["song_length"],
                ["genre_ids", "language", "artist_name"]
                )
    del item;
    gc.collect();

    # create socail network of item using dask

    # do the tag combine process

    id = "song_id"
    colList = itemDscrtAttr.columns.tolist()
    colList.remove(id)

    itemWithTag = tagCombine(itemDscrtAttr, id='song_id', tagColList=colList)

    (itemTagmatrix, itemNoAttr) = findNetwork(itemWithTag, fillnawith, split=r"&|\|")



    # if you want to do it using loop , you may set num > 2
    # if you set num = 2 ,it will do it once
    # save the social network here
    fileplace = "C:\\Users\\22560\\Desktop\\recommand Sys\\recommand Sys\\"
    LargeSparseMatrixCosine(itemTagmatrix,itemNoAttr,num=3, fileplace=fileplace,prefix="item")


    # prepare largeDisMatrix
    itemCntnueAttr.set_index("song_id", inplace=True)



    largeMatrixDis(itemCntnueAttr.values,itemHasntCntnue, num=2,
                   netFilePlace=fileplace,prefix="item")

    # set those itemHasntCntnue to have zero dis with other



    save_pickle(item_id_dict, fileplace+"item_id_dict")

    #return(item_id_dict)


if __name__ == "__main__":
    extractItemInfo()