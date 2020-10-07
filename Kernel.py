#カーネル回帰用

import numpy as np
from sklearn.model_selection import train_test_split
from itertools import combinations_with_replacement

class Kernel():

    '''カーネル回帰の準備
    入力:
        x: 入力データ
        t: 正解データ
    '''
    def __init__(self, x_train_val, x_test, t_train_val, t_test, train_indices, test_indices):

        self.beta = 1
        self.lam = 0.5
        
        self.x_train_val = x_train_val
        self.x_test = x_test
        self.t_train_val = t_train_val
        self.t_test = t_test
        self.train_indices = train_indices
        self.test_indices = test_indices

        self.dim = self.x_train_val.shape[0]


    def modeling(self):

        def kernel(xi, xj):
            return np.exp((-1) * self.beta * np.sum((xi - xj)**2))

        # グラム行列の計算
        K = np.zeros((self.dim, self.dim))
        for i, j in combinations_with_replacement(range(self.dim), 2):
            K[i][j] = kernel(self.x_train_val[i], self.x_train_val[j])
            K[j][i] = K[i][j]

        # 重みを計算
        #self.alpha = np.linalg.inv(K).dot(self.t_train_val)
        self.alpha_r = np.linalg.inv(K + self.lam * np.eye(K.shape[0])).dot(self.t_train_val)

    def prediction(self):

        def kernel(xi, xj):
            return np.exp((-1) * self.beta * np.sum((xi - xj)**2))

        # カーネル回帰
        def kernel_predict(X, x, alpha):
            Y = 0
            for i in range(len(X)):
                Y += alpha[i] * kernel(X[i], x)
            return Y

        # 回帰によって結果を予測        
        self.ar_y = np.zeros(len(self.x_test))
        for i in range(len(self.x_test)):
            self.ar_y[i] = kernel_predict(self.x_train_val, self.x_test[i], self.alpha_r)

