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
import h5py



def kernel(x):
    h = 0.5
    return np.exp(- np.sqrt(x)/h )


def yPrepareForBigData(user_num ,user_id_dict,item_id_dict,filePlace,item_id,user_id,target_id,train):

    usrList = [user_id_dict[user] for user in user_id_dict]
    usrList.sort()
    usrList = np.array(usrList)

    # select the data from train which is related with usrNowDealing
    with h5sparse.File(filePlace+"userdot_cosine.h5",'r') as user_net,\
        h5sparse.File(filePlace + "itemdot_cosine.h5",'r') as item_net,\
        h5sparse.File(filePlace + "itemdis.h5",'r') as item_dis,\
        h5sparse.File(filePlace + "userdis.h5",'r') as user_dis,\
        h5py.File(filePlace + "yPrepare.h5")  as yPrepare   :


        # # use for test
        # user_net = h5sparse.File(filePlace+"userdot_cosine.h5")
        # item_net = h5sparse.File(filePlace + "itemdot_cosine.h5")
        # item_dis = h5sparse.File(filePlace + "itemdis.h5")
        # user_dis = h5sparse.File(filePlace + "userdis.h5")
        # yPrepare =  h5py.File(filePlace + "yPrepare.h5")


        # code can be used several times

        # load the item relationship
        itemRelationship = item_net['dot_cosineData/data'].value
        itemDisRelationship = item_dis['disData/data'].value



        print("start preparing y !! , please be patient \n")
        # code related to usrNowdealing
        for userNowDealing in range(user_num):

            # obtain the useful train dataset
            usrRelationshipUsed = user_net['dot_cosineData/data'][userNowDealing:(userNowDealing+1)].\
                                            toarray().ravel()

            # get data which has relationship with userNowDealing
            usrHasRelation = usrList[usrRelationshipUsed]
            trainHasRelation = train[train[user_id].\
                isin(usrHasRelation)].sort_values(by = [item_id])


            relatedItem = trainHasRelation[item_id].values
            relatedUser = trainHasRelation[user_id].values
            relatedTarget = trainHasRelation[target_id]

            del usrHasRelation,trainHasRelation

            # get the item-train-sized  item net work
            itemToitemNetRelated = itemRelationship[:,relatedItem]

            gc.collect()



            # get the item-train-sized kernel data


            # get the item-train-sized  item  dis data
            itemToitemDisRelated = itemDisRelationship[:,relatedItem]

            gc.collect()

            # broadcast with the user-train-sized x data

            ## obtain user relationship with train
            userDisRelationship = user_dis['disData/data']\
                [userNowDealing:(userNowDealing+1)]
            userDisRelationship = userDisRelationship[:,relatedUser].todense()


            # turn it to sparse like matrix
            userDisRelationship = userDisRelationship.A.ravel()


            _idptr = itemToitemNetRelated.indptr
            _data  = userDisRelationship[itemToitemNetRelated.indices]
            _idces = itemToitemNetRelated.indices


            userDisRelationship  =  csr_matrix((_data,_idces,_idptr))


            del _idptr ; _data ; _idces
            gc.collect()

            # calculate kernel
            weight =  userDisRelationship + itemToitemDisRelated
            weight.data = kernel(weight.data)

            del userDisRelationship,itemToitemDisRelated
            gc.collect()

            weight_sum =  weight.sum(1).A.ravel()

            weight_sum_reciprocal = diags( 1/weight_sum )

            del weight_sum
            gc.collect()

            y = weight_sum_reciprocal.dot(weight).dot(relatedTarget)


            # make sure your dataset is cleaned before iteration
            if userNowDealing == 0:
                for key in yPrepare.keys():
                    del yPrepare[key]



            if userNowDealing ==0:
                yset = yPrepare.create_dataset("/yData/y",shape = (1,len(y)), maxshape=(None,len(y)) ,chunks = (1,len(y)),dtype=np.float32)
                yset[:] = y
            else:

                yset.resize(userNowDealing+1,axis = 0)
                yset[userNowDealing,:] = y


            del weight,weight_sum_reciprocal,trainHasRelation

            if(userNowDealing%10 ==0):
                print("the   ",np.round((userNowDealing+1)/(user_num-1),3 ),"  of data is prepared \n please be patient")



def yPrepareForSmallData(user_num,user_id_dict,item_id_dict,filePlace,item_id,user_id,target_id,train):


    usrList = [user_id_dict[user] for user in user_id_dict]
    usrList.sort()
    usrList = np.array(usrList)

    # select the data from train which is related with usrNowDealing
    with h5sparse.File(filePlace+"userdot_cosine.h5",'r') as user_net,\
        h5sparse.File(filePlace + "itemdot_cosine.h5",'r') as item_net,\
        h5sparse.File(filePlace + "itemdis.h5",'r') as item_dis,\
        h5sparse.File(filePlace + "userdis.h5",'r') as user_dis,\
        h5py.File(filePlace + "yPrepare.h5")  as yPrepare   :


        # # use for test
        # user_net = h5sparse.File(filePlace+"userdot_cosine.h5")
        # item_net = h5sparse.File(filePlace + "itemdot_cosine.h5")
        # item_dis = h5sparse.File(filePlace + "itemdis.h5")
        # user_dis = h5sparse.File(filePlace + "userdis.h5")
        # yPrepare =  h5py.File(filePlace + "yPrepare.h5")


        # code can be used several times

        # load the item relationship
        itemRelationship = item_net['dot_cosineData/data'].value
        itemDisRelationship = item_dis['disData/data'].value



        print("start preparing y !! , please be patient \n")
        # code related to usrNowdealing
        for userNowDealing in range(user_num):

            # obtain the useful train dataset
            usrRelationshipUsed = user_net['dot_cosineData/data'][userNowDealing:(userNowDealing+1)].\
                                            toarray().ravel()

            # get data which has relationship with userNowDealing
            usrHasRelation = usrList[usrRelationshipUsed]
            trainHasRelation = train[train[user_id].\
                isin(usrHasRelation)].sort_values(by = [item_id])



            relatedItem = trainHasRelation[item_id].values
            relatedUser = trainHasRelation[user_id].values
            relatedTarget = trainHasRelation[target_id]




            # get the item-train-sized  item net work
            itemToitemNetRelated = itemRelationship[:,relatedItem]

            gc.collect()



            # get the item-train-sized kernel data


            # get the item-train-sized  item  dis data
            itemToitemDisRelated = itemDisRelationship[:,relatedItem]

            gc.collect()

            # broadcast with the user-train-sized x data

            ## obtain user relationship with train
            userDisRelationship = user_dis['disData/data']\
                [userNowDealing:(userNowDealing+1)]
            userDisRelationship = userDisRelationship[:,relatedUser].todense()


            # turn it to sparse like matrix
            userDisRelationship = userDisRelationship.A.ravel()


            _idptr = itemToitemNetRelated.indptr
            _data  = userDisRelationship[itemToitemNetRelated.indices]
            _idces = itemToitemNetRelated.indices


            userDisRelationship  =  csr_matrix((_data,_idces,_idptr))


            del _idptr ; _data ; _idces
            gc.collect()

            # calculate kernel
            weight =  userDisRelationship + itemToitemDisRelated
            weight.data = kernel(weight.data)

            del userDisRelationship,itemToitemDisRelated
            gc.collect()

            weight_sum =  weight.sum(1).A.ravel()

            weight_sum_reciprocal = diags( 1/weight_sum )

            del weight_sum
            gc.collect()

            y = weight_sum_reciprocal.dot(weight).dot(relatedTarget)


            # make sure your dataset is cleaned before iteration
            if userNowDealing == 0:
                for key in yPrepare.keys():
                    del yPrepare[key]



            if userNowDealing ==0:
                yset = yPrepare.create_dataset("/yData/y",shape = (1,len(y)), maxshape=(None,len(y)) ,chunks = (1,len(y)),dtype=np.float32)
                yset[:] = y
            else:

                yset.resize(userNowDealing+1,axis = 0)
                yset[userNowDealing,:] = y


            del weight,weight_sum_reciprocal,trainHasRelation

            if(userNowDealing%10 ==0):
                print("the   ",np.round((userNowDealing+1)/(user_num-1),3 ),"  of data is prepared \n \n please be patient")



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


    print("train data has prepared !")

    # we are now start to prepare y
    user_num = len(user_id_dict.keys())
    yPrepareForSmallData(user_num,user_id_dict,item_id_dict,filePlace,item_id,user_id,target_id,train)



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
