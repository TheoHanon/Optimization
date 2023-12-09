from HW3_B2 import *



def long_step_method(class_A, class_B, lambda_param, nu, epsilon):

    h, c, s, t, v, mu = init(class_A, class_B)

    nA, n = class_A.vectors.shape
    nB = class_B.vectors.shape[0]

    C = c_coeff(n, nA, nB, lambda_param)

    tau = .25
    theta = .99

    ## Finding the best mu0
    mu_final = epsilon * (1 - tau) / nu

    ite = 0
    with alive_bar(10000) as bar:
        while mu > mu_final and ite < 10000:
            mu *= (1 - theta)
            h, s, t, c, v = several_newton(h, c, s, t, v, class_A, class_B, mu, lambda_param, C)
            ite += 1
            bar()


    return h, c, s, t, v


def several_newton(h, c, s, t, v, class_A, class_B, mu, lambda_param, C):

    A = class_A.vectors
    B = class_B.vectors

    nA = A.shape[0]
    nB = B.shape[0]
    n = h.size
    
    ite = 0

    
    while (ite < 1000):
    
        step = newton_step(h, c, s, t, v, class_A, class_B, mu, lambda_param, C)
        h, s, t, c, v = uptade(h, s, t, c, v, step)

        grad = C/mu +  gradF(h, s, t, c, v, n, nA, nB, class_A.vectors, class_B.vectors)

        ite += 1
        if (delta(grad, step) <=.25) :break
        
        
    return h, s, t, c, v
