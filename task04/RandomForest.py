"""
author:wjw
des: RandomForest简单实现
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier as Tree
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier as RF


class RandomForest:
    def __init__(self, n_estimators, max_depth):
        self.n_estimators = n_estimators
        self.trees = []
        self.max_depth = max_depth

    def fit(self, X, y):
        for tree_id in range(self.n_estimators):
            indexes = np.random.randint(0, X.shape[0], X.shape[0])
            random_X = X[indexes]
            random_y = y[indexes]
            tree = Tree(max_depth=3)
            tree.fit(random_X, random_y)
            self.trees.append(tree)

    def predict(self, X):
        result = []

        for x in X:
            tmp = []
            for tree in self.trees:
                tmp.append(tree.predict(x.reshape(1, -1))[0])

            result.append(np.argmax(np.bincount(tmp)))  # 返回该样本的预测结果，采取方案：多数投票
        return np.array(result)


if __name__ == '__main__':
    X, y = make_classification(n_samples=200, n_features=8, n_informative=4, random_state=0)

    RF1 = RandomForest(n_estimators=100, max_depth=3)
    RF2 = RF(n_estimators=100, max_depth=3)

    RF1.fit(X, y)
    res1 = RF1.predict(X)

    RF2.fit(X, y)
    res2 = RF2.predict(X)

    print('结果一样的比例', (np.abs(res1 - y) < 1e-5).mean())
    print('结果一样的比例', (np.abs(res2 - y) < 1e-5).mean())
