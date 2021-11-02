"""
author:wjw
分析：加权中位数的求取。
"""
from collections import defaultdict

def findMidNumber(X,weight):
    """

    :param X:输入的数据序列
    :param weight:输入序列对应长度的权重文件
    :return:输入序列中位于加权中位数的数据
    """


    w=defaultdict(list)
    for k,v in zip(X,weight):
        w[k].append(v)
    X=[(k,sum(v)) for k,v  in w.items()]
    X.sort(key=lambda x:x[0])
    n=len(X)

    presum=[0]*(1+n)
    total=0
    for i in range(1,n+1):
        total+=X[i-1][1]
        presum[i]=total

    l=0
    r=n
    while l<=r:
        mid=(l+r+1)//2
        l_w=presum[mid-1]
        r_w=total- presum[mid]
        if l_w<0.5*total and r_w<=0.5*total:
            break
        elif l_w>r_w:
            r=mid-1
        else:
            l=mid+1

    return X[mid-1][0]






