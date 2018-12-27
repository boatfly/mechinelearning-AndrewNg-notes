import numpy as np
import pandas as pd

def computeCost(X,y,theta):
    inner = np.power(((X * theta.T)-y),2)
    return np.sum(inner) / (2 * len(X))


def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost

path = 'ex1data1.txt'
data = pd.read_csv(path,header=None,names=['Population','Profit'])

data.insert(0, 'Ones', 1)

cols = data.shape[1] #获取data列数
X = data.iloc[:,0:cols-1]#X是所有行，去掉最后一列
y = data.iloc[:,cols-1:cols]#X是所有行，最后一列

X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))

cost_t = computeCost(X, y, theta)

print(cost_t)

alpha = 0.01
iters = 1000

g,cost = gradientDescent(X,y,theta,alpha,iters)

print(g)

cost_f = computeCost(X,y,g)

print(cost_f)

