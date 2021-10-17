"""
author:wjw

Description:自定义的和sklearn官方的有无区别
"""


from CART_reg import DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor as dt
from sklearn.datasets import make_regression


if __name__ == '__main__':

    X,y=make_regression(n_samples=200,n_features=8,n_informative=4,random_state=2021)

    my_cart=DecisionTreeRegressor(max_depth=3)
    my_cart.fit(X,y)
    res1=my_cart.predict(X)
    importance1=my_cart.feature_importances_

    sklearn_cart=dt(max_depth=3)
    sklearn_cart.fit(X,y)
    res2=sklearn_cart.predict(X)
    importance2=sklearn_cart.feature_importances_

    print('预测结果一样的比例',((res1-res2)<1e-8).mean())
    print('特征重要性指数一样的比例',((importance1-importance2)<1e-8).mean())