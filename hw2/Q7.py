import numpy as np
import random
import matplotlib.pyplot as plt

def sign(num, s):
	if num == 0:
		return s
	elif num > 0:
		return 1
	else:
		return -1

def gen_data(n, noise):
	x = np.random.uniform(-1,1, size=(n))
	y = np.sign(x)
	prob = np.random.uniform(0,1,n)
	y[prob<noise] *= -1
	return x, y

def hf(x, s, theta):
	if s:
		return sign(x-theta, s)
	else:
		return -sign(x-theta, s)

def err_rate(x, y, s, theta, hf):
	err = 0
	for i in range(len(x)):
		if y[i] != hf(x[i], s, theta):
			err += 1
	return err/len(x)

def decision_stump(x, y):
	#x, y = gen_data(n, noise)
	Ein = 1
	sb = True
	thetb = 0
	Theta = np.sort(x)
	S = [True, False]
	for theta in Theta:
		for s in S:
			E = err_rate(x, y, s, theta, hf)
			if E < Ein:
				Ein = E
				sb = s
				thetab = theta
	i, = np.where(Theta == thetab)
	if i[0] == 0:
		thetab = (-1+thetab)/2
	else:
		thetab = (Theta[i[0]-1]+thetab)/2

	if sb:
		Eout = 0.5+0.3*(np.abs(thetab)-1)
	else:
		Eout = 0.5-0.3*(np.abs(thetab)-1)

	return Ein, Eout

def main():
	Ein = []
	Eout = []
	Ebet = []
	n = 20 #size
	noise = 0.2
	m = 1000 #次數
	for i in range(m):
		x, y = gen_data(n, noise)
		ein, eout = decision_stump(x, y)
		Ein.append(ein)
		Eout.append(eout)
		Ebet.append(ein-eout)

	plt.hist(Ebet)
	plt.title('Ein-Eout')
	plt.show()

if __name__ == '__main__':
    main()










