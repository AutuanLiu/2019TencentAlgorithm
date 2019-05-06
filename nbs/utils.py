import modin as pd
import time


def isVaildDate(date_str):
        try:
            if ":" in date_str:
                time.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            else:
                time.strptime(date_str, "%Y-%m-%d")
            return True
        except:
            return False 


def is_more(df, field):
    """判断是否存在多值"""
    ret, l = [], len(df)
    for i in range(l):
        if len(str(df.loc[i, field]).split(',')) > 1:
            ret.append(i)
    return ret