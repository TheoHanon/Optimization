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

def gradf(x, a, b):
    return np.dot(a.T, model(x, a) - b)

def gradl1(x):
    return LAMBDA * np.sign(x)

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
    return gradf(x, a, b) + gradl1(x)


def alpha(epsilon, gradF):
    return epsilon / np.linalg.norm(gradF)**2

def sub_gradient_method(a, b, plot = False):
    
    x = np.zeros((MAXITER, a.shape[1]))
    max_g = 0
    if plot: Fk = F(x[0], a, b) * np.ones(MAXITER)

    for i in range(MAXITER-1):

        gradF = sub_gradient(x[i], a, b)
        max_g = max(max_g, np.linalg.norm(gradF))
        alphak = alpha(EPS, gradF)
        x[i+1] = x[i] - alphak * gradF


        if plot: Fk[i+1] = min(F(x[i+1], a, b), Fk[i])
    return Fk if plot else x

### Proximal gradient method ###

def prox(y, alpha):
    return np.sign(y) * np.maximum(np.zeros_like(y), abs(y) - alpha)

def proximal_gradient_method(a, b, plot = False):
    
    L = .25 * np.linalg.norm(sum(np.outer(ai, ai) for ai in a))

        
    x = np.zeros((MAXITER, a.shape[1]))
    if plot: Fk = F(x[0], a, b) * np.ones(MAXITER)

    for i in range(MAXITER-1):
        x[i+1] = prox(x[i] - 1/L * gradf(x[i], a, b), alpha = LAMBDA/L)
    
        if plot: Fk[i+1] = min(F(x[i+1], a, b), Fk[i])

    return Fk if plot else x

### Accelerated proximal gradient method ###

def accelerated_proximal_gradient_method(a, b, plot = False):
    
    L = .25 * np.linalg.norm(sum(np.outer(ai, ai) for ai in a))
  
    x = np.zeros((MAXITER, a.shape[1]))
    yold = x[0]
    told = 1
    if plot: Fk = F(x[0], a, b) * np.ones(MAXITER)

    for i in range(MAXITER-1):
     
        x[i+1] = prox(yold - 1/L * gradf(yold, a, b), LAMBDA/L)

        tnew = ( 1 + np.sqrt(1 + 4*told**2) ) / 2
        ynew = x[i+1] + (told - 1) / tnew * (x[i+1] - x[i])

        yold = ynew
        told = tnew
        if plot: Fk[i+1] = min(F(x[i+1], a, b), Fk[i])

    return Fk if plot else x

