# Re-import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from derivative import *



delta = lambda x,n: np.sqrt(np.dot(x, -n))


# Define classes A and B for storing vectors
class A:
    def __init__(self, vectors):
        self.vectors = np.array(vectors)

class B:
    def __init__(self, vectors):
        self.vectors = np.array(vectors)


def functionObjective(s, t, v, lambda_param, nA, nB):
    return lambda_param*v + 1/nA *  sum(s) + 1/nB * sum(t)


def uptade(h, s, t, c, v, step):
    nA = s.shape[0]
    nB = t.shape[0]
    n = h.shape[0]

    h = h + step[:n]
    s = s + step[n:n+nA]
    t = t + step[n+nA:n+nA+nB]
    c = c + step[n+nA+nB]
    v = v + step[-1]

    return h, s, t, c, v

def c_coeff(n, nA, nB, lambda_param):
    c = np.zeros(n+nA+nB+2)
    c[n:n+nA] = 1/nA
    c[n+nA:n+nA+nB] = 1/nB
    c[-1] = lambda_param
    return c


# Define the function to generate a starting point
def initial_feasible_point(class_A, class_B):

    A = class_A.vectors
    B = class_B.vectors 

    ## Constructing a initial feasible point

    h = np.ones(class_A.vectors.shape[1])  # Random initial h
    c = 0  # Initial guess for c
    a_max = np.max(np.dot(A, h))
    b_min = np.min(np.dot(B, h))
    s = 10*a_max + np.ones(class_A.vectors.shape[0])  # Initial s
    t = b_min / 10 + np.ones(class_B.vectors.shape[0])  # Initial t
    v = np.dot(h, h) + 1  # Initial v


    return h, c, s, t, v

# Define the Newton step as per the provided definition
def newton_step(h, c, s, t, v, class_A, class_B, mu, lambda_param, C):
    # Calculate the gradient and Hessian
    A = class_A.vectors
    B = class_B.vectors

    nA = A.shape[0]
    nB = B.shape[0]
    n = h.size

    gradBarrier = gradF(h, s, t, c, v, n, nA, nB, A, B)
    H = hessF(h, s, t, c, v, n, nA, nB, A, B)

    step = solve(H, -(gradBarrier + C/mu))
    delta_mu = delta(gradBarrier + C/mu, step)

    return step if delta_mu < 1 else step/(delta_mu + 1)

def init(class_A, class_B):
    print("Initializing...")

    nA, n = class_A.vectors.shape
    nB = class_B.vectors.shape[0]

    lambda_param = 1
    mu = 5

    h, c, s, t, v = initial_feasible_point(class_A, class_B)
    
    C = c_coeff(n, nA, nB, lambda_param)
    
    grad =  C/mu +  gradF(h, s, t, c, v, n, nA, nB, class_A.vectors, class_B.vectors)   
    step = newton_step(h, c, s, t, v, class_A, class_B, mu, lambda_param, C)

    ## Finding the best mu0

    H = hessF(h, s, t, c, v, n, nA, nB, class_A.vectors, class_B.vectors)
    d = np.linalg.solve(H, C)
    dF = gradF(h, s, t, c, v, n, nA, nB, class_A.vectors, class_B.vectors)
    mu = np.dot(-C, d) / (2 * np.dot(d, dF)) 
    
    ite = 0
    while (delta(grad, step) > .25 and ite < 1000):
        
        step = newton_step(h, c, s, t, v, class_A, class_B, mu, lambda_param, C)
        h, s, t, c, v = uptade(h, s, t, c, v, step)

        grad = C/mu +  gradF(h, s, t, c, v, n, nA, nB, class_A.vectors, class_B.vectors)
        ite += 1

    print("Done initializing...")
    return h, c, s, t, v


# Define the function to perform the optimization
def optimize(h, c, s, t, v, class_A, class_B, lambda_param, nu, epsilon):

    nA, n = class_A.vectors.shape
    nB = class_B.vectors.shape[0]

    C = c_coeff(n, nA, nB, lambda_param)

    tau = 1/4
    theta = 1/(16*np.sqrt(nu))

    ## Finding the best mu0

    H = hessF(h, s, t, c, v, n, nA, nB, class_A.vectors, class_B.vectors)
    d = np.linalg.solve(H, C)
    dF = gradF(h, s, t, c, v, n, nA, nB, class_A.vectors, class_B.vectors)

    mu = np.dot(-C, d) / (2 * np.dot(d, dF)) 
    mu_final = epsilon * (1 - tau) / nu

    ite = 0

    while mu > mu_final and ite < 10000:
        mu *= (1 - theta)
        step = newton_step(h, c, s, t, v, class_A, class_B, mu, lambda_param, C)
        h, s, t, c, v = uptade(h, s, t, c, v, step)
        ite += 1


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
a_vectors = np.random.rand(50, 2) + .2# + np.array([1, 1])
b_vectors = np.random.rand(50, 2)# + np.array([1, 5])

# Initialize class instances
class_A = A(a_vectors)
class_B = B(b_vectors)

# Set the lambda parameter for the optimization problem
lambda_param = 1
nu = 2 * len(class_A.vectors) + 2 * len(class_B.vectors) + 2
epsilon = 1e-6


h0, c0, s0, t0, v0  = init(class_A, class_B)
h, c, s, t, v       = optimize(h0, c0, s0, t0, v0, class_A, class_B, lambda_param, nu, epsilon)

print("h: ", h)
print("c: ", c)
# Plot the results
plot_data_and_separation_line(class_A.vectors, class_B.vectors, h, c)