import numpy as np

MAXITER = 500
EPS = .9
LAMBDA = 5


### Model and classifier ###

def model(x, a):
    z = np.dot(a, x)
    return 1 / (1 + np.exp(-z))

def classify(x, a):
    return (model(x, a) >= 0.5)*1.0

def F(x, a, b):
    m_x = model(x, a)  # Apply the logistic function to each example
    # Calculate the cross-entropy loss
    cross_entropy = -np.sum(b * np.log(m_x) + (1 - b) * np.log(1 - m_x))
    # Calculate the L1 norm of x
    l1_norm = LAMBDA * np.sum(np.abs(x))
    # The total cost is the sum of the cross-entropy loss and the L1 norm
    total_cost = cross_entropy + l1_norm

    return total_cost

### Sub gradient methods ###
 
def sub_gradient(x, a, b):
    errors = model(x, a) - b 
    gradf = np.dot(a.T, errors)  
    gradl1 = LAMBDA * np.sign(x) 
    gradF = gradf + gradl1

    return gradF 

def alpha(epsilon, gradF):
    return epsilon / np.linalg.norm(gradF)**2


def sub_gradient_method(a, b, plot = False):

    x = np.zeros((MAXITER, a.shape[1]))
    if plot: Fk = F(x[0], a, b) * np.ones(MAXITER)

    for i in range(MAXITER-1):
        gradF = sub_gradient(x[i], a, b)
        alphak = alpha(EPS, gradF)
        x[i+1] = x[i] - alphak * gradF
        if plot: Fk[i+1] = min(F(x[i+1], a, b), Fk[i])
    
    return Fk if plot else x

### Proximal gradient method ###

def prox(y, alpha):
    return np.sign(y) * np.maximum(np.zeros_like(y), abs(y) - alpha)

def proximal_gradient_method(a, b, plot = False):

    L = 0
    for ai in a:
        L += np.linalg.norm(np.outer(ai, ai))
        
    x = np.zeros((MAXITER, a.shape[1]))
    if plot: Fk = F(x[0], a, b) * np.ones(MAXITER)

    for i in range(MAXITER-1):
        gradf = np.dot(a.T, model(x[i], a) - b) 
        x[i+1] = prox(x[i] - 1/L * gradf, alpha = LAMBDA/L)
    
        if plot: Fk[i+1] = min(F(x[i+1], a, b), Fk[i])

    return Fk if plot else x

### Accelerated proximal gradient method ###

def accelerated_proximal_gradient_method(a, b, plot = False):

    L = 0
    for ai in a:
        L += np.linalg.norm(np.outer(ai, ai), ord = 2)

    x = np.zeros((MAXITER, a.shape[1]))
    yold = x[0]
    told = 1
    if plot: Fk = F(x[0], a, b) * np.ones(MAXITER)

    for i in range(MAXITER-1):
        gradf = np.dot(a.T, model(yold, a) - b) 
        x[i+1] = prox(yold - 1/L * gradf, LAMBDA/L)

        tnew = ( 1 + np.sqrt(1 + 4*told**2) ) / 2
        ynew = x[i+1] + (told - 1) / tnew * (x[i+1] - x[i])

        yold = ynew
        told = tnew
        if plot: Fk[i+1] = min(F(x[i+1], a, b), Fk[i])

    return Fk if plot else x

