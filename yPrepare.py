import gc
import  pandas as pd
import  os
from itemNetDisPrepare import extractItemInfo
from userNetDisPrepare import extractUserInfo




if __name__=="__main__":






    #####################   extract info from item ########################
    # first of all , you need to prepare the cleaned data in item's csv

    extractItemInfo()

    ####################    extract info from user ########################

    # first of all , you need to prepare the cleaned data in user's csv

    extractUserInfo()



    gc.collect()

    # read train data
    #train = pd.read_csv("train.csv")


    # usrNet = calculateNet(1,1,1*1e4)
    #
    # # generate two dis for test later method
    # usrDis = calculateDis(1,2,1*1e4)
    # itemDis = calculateDis(1,2,22*1e4)
    #
    # (ulist, ilist , rlist) = subtractIdx(1,180*1e4,encoded=True)
    #
