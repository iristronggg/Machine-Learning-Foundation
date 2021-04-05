import numpy as np
from matplotlib import pyplot as plt 

def sig(s):
    return 1/(np.exp(-s)+1)

def error(X, y, w):
    N, _ = X.shape
    return np.sum(np.sign(X.dot(w))!=y) / N
    
data = np.loadtxt('hw3_train.dat.txt')
X = data[:,:-1]
y = data[:,-1]
data2 = np.loadtxt('hw3_test.dat.txt')
X_test = data2[:,:-1]
y_test = data2[:,-1]
T = 2000

eout = []
eout_s = []

######GD
eta = 0.01
N, d = X.shape    
w = np.zeros(d)
for i in range(T):
    delta = np.mean((sig(-y*X.dot(w))*(-y)).reshape(N,1)*X, axis = 0)
    w = w - eta*delta

ein = error(X, y, w)
eout = error(X_test, y_test, w)

######SGD
eta = 0.001
N, d = X.shape    
w = np.zeros(d)
for i in range(T):
    n = i % N
    Xn = X[n]
    yn = y[n]
    w = w + eta*sig(-yn*Xn.dot(w))*(yn*Xn)

ein_s = error(X, y, w)
eout_s = error(X_test, y_test, w)


print('#E_in_GD', ein)
print('#E_in_SGD', ein_s)
print('#E_out_GD', eout)
print('#E_out_SGD', eout_s)



