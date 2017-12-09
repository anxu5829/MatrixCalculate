import os
# prepare User distance square
import  numpy as np
import  pandas as pd
import  os
import  gc
import collections
import h5py
import h5sparse

from usefulTool import LargeSparseMatrixCosine
from usefulTool import fillDscrtNAN
from usefulTool import fillCntnueNAN
from usefulTool import scaleCntnueVariable
from usefulTool import changeNameToID
from usefulTool import splitDF
from usefulTool import tagCombine
from usefulTool import findNetwork
from usefulTool import largeMatrixDis
from sltools    import save_pickle


def extractUserInfo():
    ####################           change dir       ######################
    os.chdir("C:\\Users\\22560\\Desktop\\recommand Sys\\recommand Sys")

    # the usual ways  to deal with date time:
    ## Suppose you have a column 'datetime' with your string, then:

    # dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    # df = pd.read_csv(infile, parse_dates=['datetime'], date_parser=dateparse)

    ##This way you can even combine multiple columns into a single datetime column, this merges a 'date' and a 'time' column into a single 'datetime' column:

    # dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    # df = pd.read_csv(infile, parse_dates={'datetime': ['date', 'time']}, date_parser=dateparse)



    # be attention of my way to deal with dates cols
    usertable = "members.csv"
    dateparse = lambda x: pd.datetime.strptime(x, '%Y%m%d')

    user = pd.read_csv(usertable,encoding = "UTF-8",
                dtype={
                    "city": str,
                    "registered_via":str,
                    "gender" : str
                },
                parse_dates= ['registration_init_time','expiration_date']
                ,date_parser=dateparse
                ,iterator= True
                       )


# use for test
    chunksize = 30000
    user  = user.get_chunk(chunksize)
    user.loc[chunksize+1,'msno'] = "special"
    user.loc[chunksize+2,'msno'] = 'special2'
# use for test end

    fillnawith = collections.OrderedDict()
    fillnawith['city'] = "no city"
    fillnawith['gender'] = 'no sex'
    fillnawith['registered_via'] = "no via"


    user = fillDscrtNAN(user,fillnawith)

    # make a continuous var for test

    user['cntinue'] =user.expiration_date\
                        -user.registration_init_time

    user.cntinue = user.cntinue.dt.days
    # other info can derived from user.continue.dt.components

    fillCntnueNAN(user,['cntinue'])
    scaleCntnueVariable(user,['cntinue'])



    user,user_id_dict = changeNameToID(user,'msno',plan='A')

    (userCntnueAttr, userDscrtAttr) = \
        splitDF(user, "msno",
                ["cntinue"],
                ["city","gender",  "registered_via"]
                )

    del user
    gc.collect()

    id = 'msno'
    colList = userDscrtAttr.columns.tolist()
    colList.remove(id)

    userWithTag = tagCombine(userDscrtAttr, id='msno', tagColList=colList)

    (userTagmatrix, userNoAttr) = findNetwork(userWithTag, fillnawith, split=r"&|\|")

    for row in userNoAttr:
        userTagmatrix[row, :] = -1

    fileplace = "C:\\Users\\22560\\Desktop\\"
    LargeSparseMatrixCosine(userTagmatrix,userNoAttr, num=20, fileplace=fileplace,prefix="user")

   # prepare largeDisMatrix
    userCntnueAttr.set_index("msno", inplace=True)

    largeMatrixDis(userCntnueAttr.values, num=20,
                   netFilePlace=fileplace ,prefix="user")

    save_pickle(user_id_dict, fileplace + "user_id_dict")
    # return(user_id_dict)