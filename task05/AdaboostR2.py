"""
author:wjw
des:adaboostR2算法的实现
为什么自己写的实现，效果不如sklearn中的呢？
"""

import numpy  as np
from median import findMidNumber
from sklearn.ensemble import AdaBoostRegressor as ABR
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

class AdaboostRegressor:
    def __init__(self,base_estimator,n_estimator):
        self.base_estimator=base_estimator
        self.n_estimator=n_estimator

        self.booster=[]
        self.weight=[]


    def fit(self,X,y,**kwargs):
        w=np.ones(X.shape[0])/X.shape[0]
        for i in range(self.n_estimator):
            cur_reg=self.base_estimator(**kwargs)
            cur_reg.fit(X,y)
            y_pred=cur_reg.predict(X)
            e=np.abs(y-y_pred)
            e/=e.max()
            err=(w*e).sum()
            beta=(1-err)/err
            alpha=np.log(beta+1e-6)
            w*=np.power(beta,1-e)
            w/=w.sum()
            self.booster.append(cur_reg)
            self.weight.append(alpha)

    def predict(self,X):
        return self._get_median_predict(X)

    def _get_median_predict(self, X):
        Y=np.array([est.predict(X) for est in self.booster]).T  #得到一个n*m的矩阵

        return np.apply_along_axis(lambda x:findMidNumber(x,self.weight),1,Y)



if __name__ == '__main__':


    X,y=make_regression(n_samples=10000,n_features=10,n_informative=5,random_state=0)

    X_train,X_valid,y_train,y_valid=train_test_split(X,y,test_size=0.3,random_state=0)

    reg=ABR(DecisionTreeRegressor(max_depth=3),n_estimators=20)

    reg.fit(X_train,y_train)
    result1=reg.predict(X_valid)

    print('sklearn中的验证集得分为: ',r2_score(y_valid,result1))


    my_reg=AdaboostRegressor(DecisionTreeRegressor,n_estimator=20)

    my_reg.fit(X_train,y_train,max_depth=3)
    result2=my_reg.predict(X_valid)
    print('自己写的AdaboostR2算法得分为: ',r2_score(y_valid,result2))














