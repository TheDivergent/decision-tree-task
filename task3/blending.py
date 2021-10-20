"""
blending的实现
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

m=3

if __name__ == '__main__':

    X,y=make_regression(n_samples=1000,n_features=8,n_informative=4,random_state=2021)
    final_X,_=make_regression(n_samples=500,n_features=8,n_informative=4,random_state=2021)

    train=pd.DataFrame(X)
    y=pd.Series(y)
    test=pd.DataFrame(final_X)


    k=4
    valid=train.iloc[[i for i in range(train.shape[0]) if i%k==0]].reset_index(drop=True)
    y_valid = y.iloc[[i for i in range(train.shape[0]) if i % k == 0]].reset_index(drop=True)
    train=train.iloc[[i for i in range(train.shape[0]) if i%k!=0]].reset_index(drop=True)
    y_train = y.iloc[[i for i in range(y.shape[0]) if i % k != 0]].reset_index(drop=True)

    final_train = pd.DataFrame(np.zeros((valid.shape[0], m)))
    final_test = pd.DataFrame(np.zeros((final_X.shape[0], m)))

    for model_id in range(m):
        model = models[model_id]

        model.fit(train, y_train)
        final_train.loc[:, model_id] = model.predict(valid)
        final_test.loc[:, model_id] = model.predict(final_X)


    final_model.fit(final_train, y_valid)
    res = final_model.predict(final_test)
    pd.Series(res).to_csv('result.csv', index=False)














