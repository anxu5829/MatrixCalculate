import gc
import numpy as np
import  pandas as pd
import  os
from sltools import load_pickle
from scipy.sparse import  vstack
from scipy.sparse import  csr_matrix
from scipy.sparse import  diags
from itemNetDisPrepare import extractItemInfo
from userNetDisPrepare import extractUserInfo
import dask.array as da
import h5sparse



def kernel(x):
    h = 0.5
    return np.exp(- np.sqrt(x)/h )


def yPrepareForBigData(userNowDealing ,user_id_dict,item_id_dict,filePlace,item_id,user_id,target_id,train):


    usrList = [user_id_dict[user] for user in user_id_dict]
    usrList.sort()
    usrList = np.array(usrList)




    # select the data from train which is related with usrNowDealing
    with h5sparse.File(filePlace+"userdot_cosine.h5") as user_net:
        # get the relationship between user u and other user:
        usrRelationship = user_net['dot_cosineData/data'][userNowDealing:(userNowDealing+1)].toarray().ravel()
        usrHasRelation = usrList[usrRelationship]
    trainHasRelation = train[train[user_id].\
            isin(usrHasRelation)].sort_values(by = [item_id])

    relatedItem = trainHasRelation[item_id].values
    relatedUser = trainHasRelation[user_id].values
    relatedTarget = trainHasRelation[target_id]

    # get the item-train-sized net work
    with h5sparse.File(filePlace+"itemdot_cosine.h5") as item_net:
        itemRelationship = item_net['dot_cosineData/data']
        itemToitemNetRelated = [itemRelationship[itemRelated:(itemRelated+1)] for itemRelated in relatedItem]
        itemToitemNetRelated = vstack(itemToitemNetRelated).transpose()


    # get the item-train-sized kernel data


    # get the item-train-sized x data

    with h5sparse.File(filePlace+"itemdis.h5") as item_dis:
        # 一次从外存中取一行的效率太低了
        # 要改
        # 对于 no tag 数据还有大问题
        itemDisRelationship = item_dis['disData/data']
        itemToitemDisRelated = [itemDisRelationship[itemDisRelated:(itemDisRelated+1)] for itemDisRelated in relatedItem]
        itemToitemDisRelated = vstack(itemToitemDisRelated).transpose()

    # broadcast with the user-train-sized x data
    with h5sparse.File(filePlace+"userdis.h5") as user_dis:
        userDisRelationship = user_dis['disData/data']\
            [userNowDealing:(userNowDealing+1)]
        userDisRelationship = userDisRelationship[:,relatedUser].todense()

    itemToitemDisRelated = itemToitemDisRelated + userDisRelationship
    itemToitemDisRelated = kernel(itemToitemDisRelated)

    weight = itemToitemNetRelated.multiply(itemToitemDisRelated)

    weight_sum  = diags(1/ weight.sum(1).A.ravel())
    weight = weight_sum @ weight

    y = weight.dot(relatedTarget.transpose())
    return(y)

def yPrepareForSmallData(userNowDealing ,user_id_dict,item_id_dict,filePlace,item_id,user_id,target_id,train):


    usrList = [user_id_dict[user] for user in user_id_dict]
    usrList.sort()
    usrList = np.array(usrList)

    # select the data from train which is related with usrNowDealing
    with h5sparse.File(filePlace+"userdot_cosine.h5") as user_net:
        # get the relationship between user u and other user:
        usrRelationship = user_net['dot_cosineData/data'][userNowDealing:(userNowDealing+1)].toarray().ravel()
        usrHasRelation = usrList[usrRelationship]
    trainHasRelation = train[train[user_id].\
            isin(usrHasRelation)].sort_values(by = [item_id])

    relatedItem = trainHasRelation[item_id].values
    relatedUser = trainHasRelation[user_id].values
    relatedTarget = trainHasRelation[target_id]

    # get the item-train-sized net work
    with h5sparse.File(filePlace+"itemdot_cosine.h5") as item_net:
        itemRelationship = item_net['dot_cosineData/data'].value
        itemToitemNetRelated = itemRelationship[:,relatedItem]
        del itemRelationship ;gc.collect()



    # get the item-train-sized kernel data

    # get the item-train-sized x data


    with h5sparse.File(filePlace+"itemdis.h5") as item_dis:
        itemDisRelationship = item_dis['disData/data'].value
        itemToitemDisRelated = itemDisRelationship[:,relatedItem]
        del itemDisRelationship ; gc.collect()

    # broadcast with the user-train-sized x data
    with h5sparse.File(filePlace+"userdis.h5") as user_dis:
        userDisRelationship = user_dis['disData/data']\
            [userNowDealing:(userNowDealing+1)]
        userDisRelationship = userDisRelationship[:,relatedUser].todense()


    # it will cause memmory error here



    userDisRelationship = userDisRelationship.A.ravel()


    _idptr = itemToitemDisRelated.indptr
    _data  = userDisRelationship[itemToitemDisRelated.indices]
    _idces = itemToitemDisRelated.indices


    userDisRelationship  =  csr_matrix((_data,_idces,_idptr))


    del _idptr ; _data ; _idces
    gc.collect()

    weight =  userDisRelationship + itemToitemDisRelated
    weight.data = kernel(weight.data)

    del userDisRelationship,itemToitemDisRelated
    gc.collect()

    weight_sum =  weight.sum(1).A.ravel()

    weight_sum_reciprocal = diags( 1/weight_sum )

    del weight_sum
    gc.collect()

    y = weight_sum_reciprocal.dot(weight).dot(trainHasRelation[target_id])

    del weight,weight_sum_reciprocal,trainHasRelation

    


def main():


    filePlace = "C:\\Users\\22560\\Desktop\\"




    user_id_dict = load_pickle(filePlace+"user_id_dict")
    item_id_dict = load_pickle(filePlace+"item_id_dict")

    gc.collect()
    # read train data
    os.chdir("C:\\Users\\22560\\Desktop\\recommand Sys\\recommand Sys")
    train = pd.read_csv("train.csv")

    user_id = "msno"
    item_id = "song_id"
    target_id  = "target"
    train[user_id] = train[user_id].map(user_id_dict)
    train[item_id] = train[item_id].map(item_id_dict)


    # bacause i just use few data of user and item,
    # i need to extract the needed row from train first
    train = train[(-pd.isna(train[user_id]))&(-pd.isna(train[item_id]))]

    train[user_id] = train[user_id].astype(np.int32)
    train[item_id] = train[item_id].astype(np.int32)

    train = train.sort_values(by = [user_id,item_id])


    train = train.groupby([user_id,item_id])['target'].sum().reset_index()

    gc.collect()


    userNowDealing = 0

    # we are now start to prepare y

    y = yPrepareForBigData(userNowDealing ,user_id_dict,item_id_dict,filePlace,item_id,user_id,target_id,train)



if __name__=="__main__":
    #extractItemInfo()
    #extractUserInfo()


    main()



# usrNet = calculateNet(1,1,1*1e4)
    #
    # # generate two dis for test later method
    # usrDis = calculateDis(1,2,1*1e4)
    # itemDis = calculateDis(1,2,22*1e4)
    #
    # (ulist, ilist , rlist) = subtractIdx(1,180*1e4,encoded=True)
    #
