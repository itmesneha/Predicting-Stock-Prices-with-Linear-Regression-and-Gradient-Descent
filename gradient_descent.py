#Gradient descent algorithm for linear regression
import matplotlib.pyplot as plt
import numpy as np
# minimize the "sum of squared errors".
#This is how we calculate and correct our error
def compute_error_for_line_given_points(b,m,points):
	totalError = 0 	#sum of square error formula
	for i in range (0, len(points)):
		x = i
		y = points[i]
		totalError += (y-(m*x + b)) ** 2
	return totalError/ float(len(points))

def step_gradient(b_current, m_current, points, learning_rate):
	#gradient descent
	b_gradient = 0
	m_gradient = 0
	N = float(len(points))
	for i in range(0, len(points)):
		x = i
		y = points[i]
		b_gradient += -(2/N) * (y - (m_current * x + b_current))
		m_gradient += -(2/N) * x * (y - (m_current * x + b_current))
	new_b = b_current - (learning_rate * b_gradient)
	new_m = m_current - (learning_rate * m_gradient)
	return [new_b,new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
	b = starting_b
	m = starting_m
	for i in range(num_iterations):
		b,m = step_gradient(b, m, np.array(points), learning_rate)
	return [b,m]

def run():
    #Step 1: Collect the data
    points = np.genfromtxt('data.csv', delimiter=',')
	#Step 2: Define our Hyperparameters
    learning_rate = 0.005 #how fast the data converge
    #y=mx+b (Slope formule)
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 2000
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    print("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print("After {0}\n iterations b = {1},\n m = {2},\n error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))
    y_predicted1=m*25+b
    y_predicted2 = m * 29 + b
    print("predicted value on feb 1st:",y_predicted1)
    print("predicted value on feb 5th:",y_predicted2)
    x_new=np.array(range(24))
    y_new=m*x_new+b
    plt.plot(x_new,y_new)
    x_plotting = list(range(23))
    y_plotting = list(points)
    plt.plot(x_plotting, y_plotting, 'ro')
    plt.show()

if __name__ == "__main__":
	run()
