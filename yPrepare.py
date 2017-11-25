# prepare S

def s_usrPrepare(u):
    # return Susr
    # Susr = ( u * omiga)
    pass


def s_itemPrepare( i ):
    # return Siem
    # Si_omiga = ( i * omiga )
    pass

def sPrepare(susr,sitem,k,mode = "u"):
    # mode 取u or i , u 表示做user 回归，从usr中取一行
    # k 表示做到第k行、列
    if (mode == "u"):
        return(sitem*susr[k,:])
    elif(mode == "i"):
        return(susr*sitem[k,:])


def yPrepare(S_ui_omiga,X_ui_omiga):
    pass