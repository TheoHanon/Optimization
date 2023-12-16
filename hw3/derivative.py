import numpy as np


### Function 
def F(h, s, t, c, v, n, nA, nB, A, B):
    return - np.sum(np.log(s - 1 - c - np.dot(A, h))) \
            - np.sum(np.log(t + c - 1 + np.dot(B, h))) \
            - np.sum(np.log(s)) -np.sum(np.log(t)) \
               -  np.log(v - np.dot(h, h))




### Derivative
def dFdh(h, s, t, c, v, n, nA, nB, A, B):
    return sum(ai / (si - 1 - c - np.dot(ai, h)) for ai, si in zip(A, s)) \
              - sum(bi / (ti + c - 1 + np.dot(bi, h)) for bi, ti in zip(B, t)) \
                + 2 * h / (v - np.dot(h, h))


def dFds(h, s, t, c, v, n, nA, nB, A, B):
    return -1 / (s - 1 - c - np.dot(A, h)) - 1 / s

def dFdt(h, s, t, c, v, n, nA, nB, A, B):
    return -1 / (t + c - 1 + np.dot(B, h)) - 1 / t

def dFdc(h, s, t, c, v, n, nA, nB, A, B):
    return sum(1 / (si - 1 - c - np.dot(ai, h)) for ai, si in zip(A, s)) \
              - sum(1 / (ti + c - 1 + np.dot(bi, h)) for bi, ti in zip(B, t))

def dFdv(h, s, t, c, v, n, nA, nB, A, B):
    return -1 / (v - np.dot(h, h)) 


### Full Gradient

def gradF(h, s, t, c, v, n, nA, nB, A, B):
    return np.block([dFdh(h, s, t, c, v, n, nA, nB, A, B), dFds(h, s, t, c, v, n, nA, nB, A, B), dFdt(h, s, t, c, v, n, nA, nB, A, B), dFdc(h, s, t, c, v, n, nA, nB, A, B), dFdv(h, c, s, t, v,n, nA, nB, A, B)])
    



### Hessians


## grad(dFdh)
def dFdhh(h, s, t, c, v, n, nA, nB, A, B):
    return sum(np.outer(ai, ai) / (si - 1 - c - np.dot(ai, h))**2 for ai, si in zip(A, s)) + \
              sum(np.outer(bi, bi) / (ti + c - 1 + np.dot(bi, h))**2 for bi, ti in zip(B, t)) + \
                2 * np.eye(n)/ (v - np.dot(h, h)) + 4 * np.outer(h, h) / (v - np.dot(h, h))**2



def dFdhs(h, s, t, c, v, n, nA, nB, A, B):
    return -A.T / (s - 1 - c - np.dot(A, h))**2


def dFdht(h, s, t, c, v, n, nA, nB, A, B):
    return B.T / (t + c - 1 + np.dot(B, h))**2

def dFdhc(h, s, t, c, v, n, nA, nB, A, B):
    return sum(ai / (si - 1 - c - np.dot(ai, h))**2 for ai, si in zip(A, s)) + \
                sum(bi / (ti + c - 1 + np.dot(bi, h))**2 for bi, ti in zip(B, t))


def dFdhv(h, s, t, c, v, n, nA, nB, A, B):
    return (- 2 * h / (v - np.dot(h, h))**2)


## grad(dFds)

def dFdss(h, s, t, c, v, n, nA, nB, A, B):
    return np.diag(1 / (s - 1 - c - np.dot(A, h))**2) + np.diag(1 / s**2)

def dFdst(h, s, t, c, v, n, nA, nB, A, B):
    return np.zeros((nA, nB))


def dFdsc(h, s, t, c, v, n, nA, nB, A, B):
    return -1 / (s - 1 - c - np.dot(A, h))**2


def dFdsv(h, s, t, c, v, n, nA, nB, A, B):
    return np.zeros(nA)



## grad(dFdt)
def dFdtt(h, s, t, c, v, n, nA, nB, A, B):
    return np.diag(1 / (t + c - 1 + np.dot(B, h))**2) + np.diag(1 / t**2)


def dFdtc(h, s, t, c, v, n, nA, nB, A, B):
    return 1 / (t + c - 1 + np.dot(B, h))**2

def dFdtv(h, s, t, c, v, n, nA, nB, A, B):
    return np.zeros(nB)

## grad(dFdc)

def dFdcc(h, s, t, c, v, n, nA, nB, A, B):
    return sum(1 / (si - 1 - c - np.dot(ai, h))**2 for ai, si in zip(A, s)) + \
                sum(1 / (ti + c - 1 + np.dot(bi, h))**2 for bi, ti in zip(B, t))

def dFdcv(h, s, t, c, v, n, nA, nB, A, B):
    return np.zeros(1)


## grad(dFdv)
def dFdvv(h, s, t, c, v, n, nA, nB, A, B):
    return 1 / (v - np.dot(h, h))**2


### FULL Hessian

def hessF(h, s, t, c, v, n, nA, nB, A, B):

    hess = np.empty((n + nA + nB + 2, n + nA + nB + 2))
    hess[:n, :n] = dFdhh(h, s, t, c, v, n, nA, nB, A, B)
    hess[:n, n:n+nA] = dFdhs(h, s, t, c, v, n, nA, nB, A, B)
    hess[:n, n+nA:n+nA+nB] = dFdht(h, s, t, c, v, n, nA, nB, A, B)
    hess[:n, n+nA+nB] = dFdhc(h, s, t, c, v, n, nA, nB, A, B)
    hess[:n, n+nA+nB+1] = dFdhv(h, s, t, c, v, n, nA, nB, A, B)

    hess[n:n+nA, :n] = dFdhs(h, s, t, c, v, n, nA, nB, A, B).T
    hess[n:n+nA, n:n+nA] = dFdss(h, s, t, c, v, n, nA, nB, A, B)
    hess[n:n+nA, n+nA:n+nA+nB] = dFdst(h, s, t, c, v, n, nA, nB, A, B)
    hess[n:n+nA, n+nA+nB] = dFdsc(h, s, t, c, v, n, nA, nB, A, B)
    hess[n:n+nA, n+nA+nB+1] = dFdsv(h, s, t, c, v, n, nA, nB, A, B)

    hess[n+nA:n+nA+nB, :n] = dFdht(h, s, t, c, v, n, nA, nB, A, B).T
    hess[n+nA:n+nA+nB, n:n+nA] = dFdst(h, s, t, c, v, n, nA, nB, A, B).T
    hess[n+nA:n+nA+nB, n+nA:n+nA+nB] = dFdtt(h, s, t, c, v, n, nA, nB, A, B)
    hess[n+nA:n+nA+nB, n+nA+nB] = dFdtc(h, s, t, c, v, n, nA, nB, A, B)
    hess[n+nA:n+nA+nB, n+nA+nB+1] = dFdtv(h, s, t, c, v, n, nA, nB, A, B)

    hess[n+nA+nB, :n] = dFdhc(h, s, t, c, v, n, nA, nB, A, B).T
    hess[n+nA+nB, n:n+nA] = dFdsc(h, s, t, c, v, n, nA, nB, A, B).T
    hess[n+nA+nB, n+nA:n+nA+nB] = dFdtc(h, s, t, c, v, n, nA, nB, A, B).T
    hess[n+nA+nB, n+nA+nB] = dFdcc(h, s, t, c, v, n, nA, nB, A, B)
    hess[n+nA+nB, n+nA+nB+1] = dFdcv(h, s, t, c, v, n, nA, nB, A, B)

    hess[n+nA+nB+1, :n] = dFdhv(h, s, t, c, v, n, nA, nB, A, B).T
    hess[n+nA+nB+1, n:n+nA] = dFdsv(h, s, t, c, v, n, nA, nB, A, B).T
    hess[n+nA+nB+1, n+nA:n+nA+nB] = dFdtv(h, s, t, c, v, n, nA, nB, A, B).T
    hess[n+nA+nB+1, n+nA+nB] = dFdcv(h, s, t, c, v, n, nA, nB, A, B).T
    hess[n+nA+nB+1, n+nA+nB+1] = dFdvv(h, s, t, c, v, n, nA, nB, A, B)

    return hess




# n = 2
# nA = 1
# nB = 1
# A = np.zeros((nA, n))
# B = np.zeros((nB, n))

# h = np.ones(n)
# s = 100*np.ones(nA)
# t = 100*np.ones(nB)
# c = 1
# v = 1

# print(hessF(h, s, t, c, v, n, nA, nB, A, B))