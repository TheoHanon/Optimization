# Re-import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from derivative import *

# Define classes A and B for storing vectors
class A:
    def __init__(self, vectors):
        self.vectors = np.array(vectors)

class B:
    def __init__(self, vectors):
        self.vectors = np.array(vectors)


def function(s, t, v, lambda_param, nA, nB):
    return lambda_param*v + 1/nA *  sum(s) + 1/nB * sum(t)

def c_coeff(n, nA, nB, lambda_param):
    c = np.zeros(n+nA+nB+2)
    c[n:n+nA] = 1/nA
    c[n+nA:n+nA+nB] = 1/nB
    c[-1] = lambda_param
    return c


# Define the function to generate a starting point
def initial_feasible_point(class_A, class_B):

    h = np.array([1.0, 2.0])  # Random initial h
    c = -4  # Initial guess for c
    s = np.ones(class_A.vectors.shape[0])  # Initial s
    t = np.ones(class_B.vectors.shape[0])  # Initial t
    v = np.linalg.norm(h) + 1  # Initial v

    
    return h, c, s, t, v

# Define the Newton step as per the provided definition
def newton_step(h, c, s, t, v, class_A, class_B, mu, lambda_param):
    # Calculate the gradient and Hessian
    A = class_A.vectors
    B = class_B.vectors
    nA = A.shape[0]
    nB = B.shape[0]
    n = h.size

    gradBarrier = gradF(h, s, t, c, v, n, nA, nB, A, B)
    C = c_coeff(n, nA, nB, lambda_param)
    H = hessF(h, s, t, c, v, n, nA, nB, A, B)

    step = solve(H, -(gradBarrier + C/mu))
    delta = np.sqrt(np.dot(-step, gradBarrier+C/mu))

    return step if delta < 1 else step/(delta + 1)

def init(class_A, class_B):
    print("Initializing...")
    h = np.array([1.0, 1.0])  # Random initial h
    c = 0  # Initial guess for c
    s = 1000*np.ones(class_A.vectors.shape[0])  # Initial s
    t = 1000*np.ones(class_B.vectors.shape[0])  # Initial t
    v = np.dot(h, h) + 1  # Initial v

    nA = class_A.vectors.shape[0]
    nB = class_B.vectors.shape[0]
    n = h.size

    # Set the lambda parameter for the optimization problem
    lambda_param = 1
    mu = 5
    # Calculate the barrier parameter nu


    # Set target accur

    delta = lambda x,n: np.sqrt(np.dot(x, -n))
    C = c_coeff(n, nA, nB, lambda_param)
    
    grad =  C/mu +  gradF(h, s, t, c, v, n, nA, nB, class_A.vectors, class_B.vectors)   
    step = newton_step(h, c, s, t, v, class_A, class_B, mu, lambda_param)
    ite = 0
    while (delta(grad, step) > .25 and ite < 1000):
        
        step = newton_step(h, c, s, t, v, class_A, class_B, mu, lambda_param)
        h += step[:n]
        s += step[n:n+nA]
        t += step[n+nA:n+nA+nB]
        c += step[n+nA+nB]
        v += step[-1]
        grad = C/mu +  gradF(h, s, t, c, v, n, nA, nB, class_A.vectors, class_B.vectors)
        ite += 1
    print("Done initializing...")
    return h, c, s, t, v





# Define the function to perform the optimization
def optimize(h, c, s, t, v, class_A, class_B, lambda_param, nu, epsilon):

    nA = class_A.vectors.shape[0]
    nB = class_B.vectors.shape[0]
    n = h.size

    tau = 1/4
    theta = 1/(16*np.sqrt(nu))
    mu = 5
    mu_final = epsilon * (1 - tau) / nu

    ite = 0

    while mu > mu_final: #and ite < 10000
        mu *= (1 - theta)
        step = newton_step(h, c, s, t, v, class_A, class_B, mu, lambda_param)
        ite += 1

        h += step[:n]
        s += step[n:n+nA]
        t += step[n+nA:n+nA+nB]
        c += step[n+nA+nB]
        v += step[-1]
        
    return h, c, s, t, v

# Define the function to plot data and separation line
def plot_data_and_separation_line(a_vectors, b_vectors, h, c):
    plt.scatter(a_vectors[:, 0], a_vectors[:, 1], color='blue', label='Class A')
    plt.scatter(b_vectors[:, 0], b_vectors[:, 1], color='red', label='Class B')
    x_values = np.linspace(0, 3, 100)
    # print(x_values)
    y_values = (-h[0] / h[1]) * x_values - c / h[1]
    plt.plot(x_values, y_values, label='Separation Line')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('2D Data and Separation Line')
    plt.legend()
    plt.show()

# Generate some example 2D data
a_vectors = np.random.rand(100, 2) +.2# + np.array([1, 1])
b_vectors = np.random.rand(100, 2)# + np.array([1, 5])

# Initialize class instances
class_A = A(a_vectors)
class_B = B(b_vectors)

# Set the lambda parameter for the optimization problem
lambda_param = 1
nu = 2 * len(class_A.vectors) + 2 * len(class_B.vectors) + 2
epsilon = 1e-6
# Set target accuracy


# Obtain an initial feasible point
h0, c0, s0, t0, v0  = init(class_A, class_B)
h, c, s, t, v       = optimize(h0, c0, s0, t0, v0, class_A, class_B, lambda_param, nu, epsilon)

print("h: ", h)
print("c: ", c)
# Plot the results
plot_data_and_separation_line(class_A.vectors, class_B.vectors, h, c)