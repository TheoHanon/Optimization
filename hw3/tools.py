from sklearn.decomposition import PCA
import numpy as np
from derivative import *
import numpy as np
from scipy.linalg import solve



delta = lambda H,n: np.sqrt(np.dot(n, np.dot(H, n)))


# Define classes A and B for storing vectors
class A:
    def __init__(self, vectors):
        self.vectors = np.array(vectors)

class B:
    def __init__(self, vectors):
        self.vectors = np.array(vectors)


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
    
    try:
        step = solve(H, -(gradBarrier + C/mu), assume_a='sym', overwrite_a=True, overwrite_b=True)
        delta_mu = delta(H, step)
    except:
        step = np.zeros(n+nA+nB+2)
        delta_mu = 0
      
    return step if delta_mu < 1 else step/(delta_mu + 1)


def init(class_A, class_B, tau):
   
    nA, n = class_A.vectors.shape
    nB = class_B.vectors.shape[0]

    lambda_param = 1
    h, c, s, t, v = initial_feasible_point(class_A, class_B)
    
    C = c_coeff(n, nA, nB, lambda_param)

    ## Finding the best mu0
    H = hessF(h, s, t, c, v, n, nA, nB, class_A.vectors, class_B.vectors)
    d = solve(H, C, assume_a='sym')
    dF = gradF(h, s, t, c, v, n, nA, nB, class_A.vectors, class_B.vectors)

    mu = np.dot(-C, d) / (np.dot(d, dF))
    
    grad =  C/mu +  gradF(h, s, t, c, v, n, nA, nB, class_A.vectors, class_B.vectors)   
    step = newton_step(h, c, s, t, v, class_A, class_B, mu, lambda_param, C)

    ite = 0

    while (delta(H, step) > tau):
        
        step = newton_step(h, c, s, t, v, class_A, class_B, mu, lambda_param, C)
        h, s, t, c, v = uptade(h, s, t, c, v, step)
        H = hessF(h, s, t, c, v, n, nA, nB, class_A.vectors, class_B.vectors)
        
        ite += 1

    return h, c, s, t, v, mu




def several_newton(h, c, s, t, v, class_A, class_B, mu, lambda_param, C, tau):

    A = class_A.vectors
    B = class_B.vectors

    nA = A.shape[0]
    nB = B.shape[0]
    n = h.size
    
    ite = 0

    
    while (ite < 1000):
    
        step = newton_step(h, c, s, t, v, class_A, class_B, mu, lambda_param, C)
        h, s, t, c, v = uptade(h, s, t, c, v, step)
        H = hessF(h, s, t, c, v, n, nA, nB, A, B)
    
        ite += 1
        if (delta(H, step) <=tau) : break
        
    return h, s, t, c, v





def apply_pca(train_data, test_data, n_components):
    pca = PCA(n_components=n_components)
    train_pca = pca.fit_transform(train_data)
    test_pca = pca.transform(test_data)
    return train_pca, test_pca


def import_data(use_pca = False, n_pca = 5):

    data_path = "./"

    # Load data
    train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter=",")
    test_data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",")

    # Normalize
    fac = 0.99 / 255
    train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
    test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01

    train_labels = np.asfarray(train_data[:, :1])
    test_labels = np.asfarray(test_data[:, :1])

    # Function to apply PCA


    # Apply PCA if chosen
    if use_pca:  # 'use_pca' should be a boolean variable set by the user
        train_imgs, test_imgs = apply_pca(train_imgs, test_imgs, n_pca)

    return train_imgs, test_imgs, train_labels, test_labels


def get_class(Nimage, imgs, labels):
    vector_A = imgs[np.where(labels == 0)[0]]
    vector_B = np.concatenate([imgs[np.where(labels == i)[0]][:(Nimage)] for i in range(1, 10)])
    vector_A = vector_A[:(Nimage*9)]

    class_A = A(vector_A)
    class_B = B(vector_B)

    return class_A, class_B


def classify(x, h, c):
    return 0 if np.dot(h, x,) + c < -1 else 1
