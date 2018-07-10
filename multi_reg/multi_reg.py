import numpy as np
import matplotlib.pyplot as plt

test_data = np.genfromtxt('test.csv',dtype=np.float64,delimiter=",",usecols=(2,3,4,5,6),skip_header=1)

test_data = (test_data - test_data.mean())/ test_data.std()

m = test_data.shape[0]

x = np.asmatrix(test_data[:,1:])
x = np.asmatrix(np.append(np.ones([x.shape[0],1],dtype=np.float64),x,axis=1))

y = np.asmatrix(test_data[:,0]).T

theta = np.matrix([[1],[1],[1],[1],[1]])

print(theta.shape)
def computeCost(x,y,theta,m):
		return (((x * theta - y).T) * (x * theta - y))/(2*m)

print(computeCost(x,y,theta,m))
iters = 4000
alpha = 0.001

J = []

for i in range(iters):
		theta = theta - ((alpha*(x.T*(x*theta - y)))/m)
		if(i%1 == 0):
				J.append(computeCost(x,y,theta,m))

plt.subplot(211)
plt.scatter([x[:,0]],[y])
plt.xlabel('No. of bedrooms')
plt.ylabel('Price of the house')
plt.title('No. of bedrooms vs price')
plt.subplot(212)
plt.plot(range(4000),np.squeeze(np.array(J)))
plt.xlabel('No. of iterations')
plt.ylabel('Cost function')
plt.title('iterations vs cost function')
plt.show()

print(computeCost(x,y,theta,m))
