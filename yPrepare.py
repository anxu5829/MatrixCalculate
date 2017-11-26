# prepare User distance square
import numpy as np

def calculateDis(ObjectAttr,attrNum,ObjectNum):
    """

    :param ObjectAttr: must be a matrix has a shape like (ObjectNum *　attrNum)
    :param attrNum: the  number of continuous attr a usr have
    :param ObjectNum: the number of usr we observed
    :return:

    ObjectAttr matrix is ObjectNum * attrNum matrix
    target is to return a Object *　Object matrix  ， saving value of
    square  of distance  of each Object

    """
    #　return a test matrix for other
    return np.ones((ObjectNum,ObjectNum))



def calculateNet(ObjectAttr,attrNum,ObjectNum):
    """

    :param ObjectAttr: must be a matrix has a shape like (ObjectNum *　attrNum)
    :param attrNum: the  number of categorical attr a usr have
    :param ObjectNum: the number of usr we observed
    :return:

    ObjectAttr matrix is ObjectNum * attrNum matrix
    target is to return a Object *　Object matrix  ， saving value of
    the social net work of each Object

    """
    #　return a test matrix for other
    return np.ones((ObjectNum,ObjectNum))



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


# 可再进一步开发q的回归工具




if __name__=="__main__":
    # generate two dis for test later method
    usrDis = calculateDis(1,2,1*1e4)
    itemDis = calculateDis(1,2,22*1e4)
    usrNet = calculateNet(1,1,1*1e4)
    itemNet = calculateNet(1,1,22*1e4)
    (ulist, ilist , rlist) = subtractIdx(1,180*1e4,encoded=True)


