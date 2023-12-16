from HW3_B2 import *


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
        if (delta(H, step) <=tau) :break
        
        
    return h, s, t, c, v
