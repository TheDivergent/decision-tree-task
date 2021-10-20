"""
stacking的简易实现.
"""

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import  LinearRegression
from sklearn.datasets import make_regression
from sklearn.svm import  LinearSVR

import numpy as np
import pandas as pd

m1=KNeighborsRegressor()
m2=DecisionTreeRegressor()
m3=LinearRegression()

models=[m1,m2,m3]

final_model=LinearSVR()

k,m=4,3

if __name__ == '__main__':

    X,y=make_regression(n_samples=1000,n_features=8,n_informative=4,random_state=2021)

    final_X,_=make_regression(n_samples=500,n_features=8,n_informative=4,random_state=2021)


    final_train=pd.DataFrame(np.zeros((X.shape[0],m)))
    final_test=pd.DataFrame(np.zeros((final_X.shape[0],m)))

    kf=KFold(n_splits=k)

    for model_id in range(m):
        model=models[model_id]
        for train_part_index,valid_part_index in kf.split(X,y):
            X_train,y_train=X[train_part_index],y[train_part_index]
            X_valid,y_valid=X[valid_part_index],y[valid_part_index]
            model.fit(X_train,y_train)
            final_train.loc[valid_part_index,model_id]=model.predict(X_valid)
            final_test.loc[:,model_id]+=model.predict(final_X)
        final_test.loc[:,model_id]/=k #原来的代码有问题k折交叉验证应该是除以k

    final_model.fit(final_train,y)
    res=final_model.predict(final_test)
    pd.Series(res).to_csv('result.csv',index=False)







