import numpy as np
import matplotlib.pyplot as plt


def main():
		test_data = np.genfromtxt('test.csv',delimiter=",",skip_header=1);
		
		theta_0 = -0.5
		theta_1 = 1
		
		x = test_data[:,0]
		y = test_data[:,1]

		m = len(x)
		alpha = 0.0001
		iters = 1000

		j1 = computeCost(x,y,theta_0,theta_1,m)
		print('Intial Cost function =',j1)

		Jvals = []

		for i in range(iters):
				temp0 = theta_0 - (alpha*sum(theta_0 + theta_1*x - y))/m
				temp1 = theta_1 - (alpha*sum((theta_0 + theta_1*x - y)*x))/m
				theta_0 = temp0
				theta_1 = temp1
				J1 = computeCost(x,y,theta_0,theta_1,m)
				Jvals.append(J1)

		print('theta0 =',theta_0)
		print('theta1 =',theta_1)
		
		J = computeCost(x,y,theta_0,theta_1,m)
		print('Cost Function J() =',J)
		print('Test value =',theta_0 + theta_1*24)

		showGraph(x,y,iters,Jvals,theta_0,theta_1)

def computeCost(x,y,_t0,_t1,m):
		return sum((_t0 + _t1*x - y) ** 2)/(2*m)

def showGraph(x,y,_iters,J,theta0,theta1):

		plt.subplot(211)
		plt.plot(x[0:2],y[0:2],color='r')
		plt.scatter(theta0 + theta1*x,y)
		plt.xlabel('Size (in sqr fts)');
		plt.ylabel('Cost (in Rs)')
		plt.title('Costs of plots')

		plt.subplot(212)
		plt.xlabel('Cost function (J)')
		plt.ylabel('number of iterations')
		plt.title('Relationship of Cost function with iters')
		plt.yscale('log')
		plt.scatter(range(_iters),J)

		plt.show()

if __name__ == "__main__":
		main()
