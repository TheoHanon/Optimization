import numpy as np




### Function 
def F(h, c, s, t, v, nA, nB, A, B):
    return - np.sum(np.log(s - 1 - c - np.dot(A, h))) \
            - np.sum(np.log(t + c - 1 + np.dot(B, h))) \
            - np.sum(np.log(s)) -np.sum(np.log(t)) \
               -  np.sum(v**2 - np.dot(h, h)) - np.log(v)




### Derivative
def dFdh(h, c, s, t, v, n, nA, nB, A, B):
    return sum(ai / (si - 1 - c - np.dot(ai, h)) for ai, si in zip(A, s)) - \
              sum(bi / (ti + c - 1 + np.dot(bi, h)) for bi, ti in zip(B, t)) + \
                2 * h / (v**2 - np.dot(h, h))


def dFds(h, c, s, t, v, n, nA, nB, A, B):
    return -1 / (s - 1 - c - np.dot(A, h)) - 1 / s

def dFdt(h, c, s, t, v, n, nA, nB, A, B):
    return -1 / (t + c - 1 + np.dot(B, h)) - 1 / t

def dFdc(h, c, s, t, v,n, nA, nB, A, B):
    return sum(1 / (si - 1 - c - np.dot(ai, h)) for ai, si in zip(A, s)) - \
              sum(1 / (ti + c - 1 + np.dot(bi, h)) for bi, ti in zip(B, t))


def dFdv(h, c, s, t, v, n, nA, nB, A, B):
    return -2 * v / (v**2 - np.dot(h, h)) - 1/v 


### Full Gradient

def gradF(h, c, s, t, v,n, nA, nB, A, B):
    return np.array([*dFdh(h, c, s, t, v), *dFds(h, c, s, t, v), *dFdt(h, c, s, t, v), dFdc(h, c, s, t, v), dFdv(h, c, s, t, v)])



### Hessians


## grad(dFdh)
def dFdhh(h, c, s, t, v, n, nA, nB, A, B):
    return sum(np.outer(ai, ai) / (si - 1 - c - np.dot(ai, h))**2 for ai, si in zip(A, s)) + \
              sum(np.outer(bi, bi) / (ti + c - 1 + np.dot(bi, h))**2 for bi, ti in zip(B, t)) + \
                2 * np.eye(n)/ (v**2 - np.dot(h, h)) - 4 * np.outer(h, h) / (v**2 - np.dot(h, h))**2



def dFdhs(h, c, s, t, v, n, nA, nB, A, B):
    return -A.T / (s - 1 - c - np.dot(A, h))**2


def dFdht(h, c, s, t, v, n, nA, nB, A, B):
    return np.zeros((n, nB))

def dFdhc(h, c, s, t, v, n, nA, nB, A, B):
    return sum(ai / (si - 1 - c - np.dot(ai, h))**2 for ai, si in zip(A, s)) + \
                sum(bi / (ti + c - 1 + np.dot(bi, h))**2 for bi, ti in zip(B, t))


def dFdhv(h, c, s, t, v, n, nA, nB, A, B):
    return (- 4 * h * v / (v**2 - np.dot(h, h))**2)


## grad(dFds)

def dFdss(h, c, s, t, v, n, nA, nB, A, B):
    return np.diag(1 / (s - 1 - c - np.dot(A, h))**2) - np.diag(1 / s**2)

def dFdst(h, c, s, t, v, n, nA, nB, A, B):
    return np.zeros((nA, nB))


def dFdsc(h, c, s, t, v, n, nA, nB, A, B):
    return -1 / (s - 1 - c - np.dot(A, h))**2


def dFdsv(h, c, s, t, v, n, nA, nB, A, B):
    return np.zeros(nA)



## grad(dFdt)
def dFdtt(h, c, s, t, v, n, nA, nB, A, B):
    return np.diag(1 / (t + c - 1 + np.dot(B, h))**2) + np.diag(1 / t**2)


def dFdtc(h, c, s, t, v, n, nA, nB, A, B):
    return 1 / (t + c - 1 + np.dot(B, h))**2

def dFdtv(h, c, s, t, v, n, nA, nB, A, B):
    return np.zeros(nB)

## grad(dFdc)

def dFdcc(h, c, s, t, v, n, nA, nB, A, B):
    return sum(1 / (si - 1 - c - np.dot(ai, h))**2 for ai, si in zip(A, s)) + \
                sum(1 / (ti + c - 1 + np.dot(bi, h))**2 for bi, ti in zip(B, t))

def dFdcv(h, c, s, t, v, n, nA, nB, A, B):
    return np.zeros(1)


## grad(dFdv)
def dFdvv(h, c, s, t, v, n, nA, nB, A, B):
    return -2 / (v**2 - np.dot(h, h)) +4 * v / (v**2 - np.dot(h, h))**2 + 1/v**2


### FULL Hessian

def hessF(h, c, s, t, v, n, nA, nB, A, B):
    dFF = np.empty((n+nA+nB+2, n+nA+nB+2))

    dFF[:n, :n] = dFdhh(h, c, s, t, v, n, nA, nB, A, B)
    dFF[:n, n:n+nA] = dFdhs(h, c, s, t, v, n, nA, nB, A, B)
    dFF[:n, n+nA:n+nA+nB] = dFdht(h, c, s, t, v, n, nA, nB, A, B)
    dFF[:n, n+nA+nB] = dFdhc(h, c, s, t, v, n, nA, nB, A, B)
    dFF[:n, n+nA+nB+1] = dFdhv(h, c, s, t, v, n, nA, nB, A, B)

    dFF[n:n+nA, :n] = dFdhs(h, c, s, t, v, n, nA, nB, A, B).T
    dFF[n:n+nA, n:n+nA] = dFdss(h, c, s, t, v, n, nA, nB, A, B)
    dFF[n:n+nA, n+nA:n+nA+nB] = dFdst(h, c, s, t, v, n, nA, nB, A, B)
    dFF[n:n+nA, n+nA+nB] = dFdsc(h, c, s, t, v, n, nA, nB, A, B)
    dFF[n:n+nA, n+nA+nB+1] = dFdsv(h, c, s, t, v, n, nA, nB, A, B)

    dFF[n+nA:n+nA+nB, :n] = dFdht(h, c, s, t, v, n, nA, nB, A, B).T
    dFF[n+nA:n+nA+nB, n:n+nA] = dFdst(h, c, s, t, v, n, nA, nB, A, B).T
    dFF[n+nA:n+nA+nB, n+nA:n+nA+nB] = dFdtt(h, c, s, t, v, n, nA, nB, A, B)
    dFF[n+nA:n+nA+nB, n+nA+nB] = dFdtc(h, c, s, t, v, n, nA, nB, A, B)
    dFF[n+nA:n+nA+nB, n+nA+nB+1] = dFdtv(h, c, s, t, v, n, nA, nB, A, B)

    dFF[n+nA+nB, :n] = dFdhc(h, c, s, t, v, n, nA, nB, A, B)
    dFF[n+nA+nB, n:n+nA] = dFdsc(h, c, s, t, v, n, nA, nB, A, B)
    dFF[n+nA+nB, n+nA:n+nA+nB] = dFdtc(h, c, s, t, v, n, nA, nB, A, B)
    dFF[n+nA+nB, n+nA+nB] = dFdcc(h, c, s, t, v, n, nA, nB, A, B)
    dFF[n+nA+nB, n+nA+nB+1] = dFdcv(h, c, s, t, v, n, nA, nB, A, B)

    dFF[n+nA+nB+1, :n] = dFdhv(h, c, s, t, v, n, nA, nB, A, B)
    dFF[n+nA+nB+1, n:n+nA] = dFdsv(h, c, s, t, v, n, nA, nB, A, B)
    dFF[n+nA+nB+1, n+nA:n+nA+nB] = dFdtv(h, c, s, t, v, n, nA, nB, A, B)
    dFF[n+nA+nB+1, n+nA+nB] = dFdcv(h, c, s, t, v, n, nA, nB, A, B)
    dFF[n+nA+nB+1, n+nA+nB+1] = dFdvv(h, c, s, t, v, n, nA, nB, A, B)

    return dFF




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

# print(hessF(h, c, s, t, v, n, nA, nB, A, B))