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

ein = []
ein_s = []

plt.figure()
######GD
eta = 0.01
N, d = X.shape    
w = np.zeros(d)
for i in range(T):
    delta = np.mean((sig(-y*X.dot(w))*(-y)).reshape(N,1)*X, axis = 0)
    w = w - eta*delta
    ein.append(error(X, y, w))

######SGD
eta = 0.001
N, d = X.shape    
w = np.zeros(d)
for i in range(T):
    n = i % N
    Xn = X[n]
    yn = y[n]
    w = w + eta*sig(-yn*Xn.dot(w))*(yn*Xn)
    ein_s.append(error(X, y, w))


I = []
for i in range(T):
	a = i+1
	I.append(a)

plt.plot(I, ein, 'r', label='E_in_GD')
plt.plot(I, ein_s, 'g', label='E_in_SGD')
plt.legend(loc='upper right')
plt.show()



