import numpy as np
import matplotlib.pyplot as plt

def convert(name):
		if name == 'Iris-setosa':
				return 0
		else:
				return 1

X = np.genfromtxt('Iris.csv',delimiter=',',skip_header=1,usecols=(1,2,3,4))[0:100]
Y = np.genfromtxt('Iris.csv',delimiter=',',skip_header=1,usecols=5,dtype=str)[0:100]

X = np.asmatrix(np.append(np.ones([X.shape[0],1],dtype=np.float64),X,axis=1))
Y = np.asmatrix(list(map(convert,Y)))

m = X.shape[0]
iters = 3000
alpha = 0.1
theta = np.matrix([1,1,1,1,1])
cost = []

def Cost(x,y,theta,m):
		h = 1/(1 + np.exp(-(theta * x.T)))
		sum = 0
		cost = []
		for i in range(m):
				cost = np.append(cost,(y[0,i]*(np.log(h[0,i])) + ((1-y[0,i])*(np.log(1 - h[0,i])))))
				sum += cost[i]
		return -sum/m
		
def GradientDescent(x,y,theta,alpha,iters,m):
		cost = []
		for i in range(iters):
				h = 1/(1 + np.exp(-(theta * x.T)))
				theta = theta - (alpha*((h - y)*x))/m
				if(iters % 100 == 0):
						J = Cost(x,y,theta,m)
						cost = np.append(cost,J)
		return theta,cost

theta,cost = GradientDescent(X,Y,theta,alpha,iters,m)

print(Cost(X,Y,theta,m))

plt.plot(range(iters),cost)
plt.show()
