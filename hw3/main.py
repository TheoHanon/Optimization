from sklearn.decomposition import PCA
import numpy as np
from derivative import *
from HW3_B2 import *
from HW3_B4 import several_newton
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from progress.bar import Bar
from alive_progress import alive_bar
from sklearn.metrics import accuracy_score
from tools import *
import time


def long_step_method(class_A, class_B, lambda_param, nu, epsilon, theta):

    h, c, s, t, v, mu = init(class_A, class_B)

    nA, n = class_A.vectors.shape
    nB = class_B.vectors.shape[0]

    C = c_coeff(n, nA, nB, lambda_param)

    tau = .10

    mu_final = epsilon * (1 - tau) / nu
    nmax = int(np.ceil(np.log(mu_final/mu)/np.log(1 - theta)))
    ite = 0
    with alive_bar(nmax) as bar:
        while mu > mu_final:
            mu *= (1 - theta)
            h, s, t, c, v = several_newton(h, c, s, t, v, class_A, class_B, mu, lambda_param, C, tau)
            ite += 1
            bar()


    return h, c, s, t, v, ite


def short_step_method(class_A, class_B, lambda_param, nu, epsilon):

    nA, n = class_A.vectors.shape
    nB = class_B.vectors.shape[0]

    C = c_coeff(n, nA, nB, lambda_param)

    tau = 1/4
    theta = 1/(16*np.sqrt(nu))

    h, c, s, t, v, mu = init(class_A, class_B)
    mu_final = epsilon * (1 - tau) / nu
    nmax = int(np.ceil(np.log(mu_final/mu)/np.log(1 - theta)))
    ite = 0
    with alive_bar(nmax) as bar:
        while mu > mu_final:
            mu *= (1 - theta)
            step = newton_step(h, c, s, t, v, class_A, class_B, mu, lambda_param, C)
            h, s, t, c, v = uptade(h, s, t, c, v, step)
            ite += 1
            bar()

    return h, c, s, t, v, ite




## Import data 1

# NIMAGE = 100
# train_imgs, test_imgs, train_labels, test_labels = import_data(use_pca = True, n_pca = 5)
# class_A, class_B = get_class(NIMAGE, train_imgs, train_labels)


# ## Lambdas

lambda_param = [.01, .02, .03, .05, .1, .4, .5, 1, 10, 50, 100, 1000]
# nu = 2 * len(class_A.vectors) + 2 * len(class_B.vectors) + 2
# epsilon = 1e-6
# acc = [] 

# for lam in lambda_param:
#     h, c, s, t, v, ite = long_step_method(class_A, class_B, lam, nu, epsilon, .95)

#     ypred = np.array([classify(x, h, c) for x in train_imgs])
#     ytrue = (train_labels > 0)*1.0
#     acc.append(accuracy_score(ytrue, ypred))


acc=  [0.3755833333333333, 0.3862, 0.3986, 0.42961666666666665 ,0.8497, 0.9101166666666667, 0.9183166666666667, 0.942, 0.9517, 0.9517 ,0.9517, 0.9012833333333333]
print("acc : ", *acc)

plt.figure()
plt.plot(lambda_param, acc, "^-", color = "black")
plt.xlabel("Lambda")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Lambda")
# plt.savefig("lambdas_acc.pdf")
plt.show()

plt.figure()   
plt.plot(lambda_param, acc, "^-", color = "black")
plt.xlabel("Lambda")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Lambda")
plt.xscale("log")
# plt.savefig("lambdas_acc_log.pdf")
plt.show()


### Import data 2

train_imgs, test_imgs, train_labels, test_labels = import_data(use_pca = False)

nImage = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20]
# timeCPU = []
# nIte = []

# for n in nImage:

#     class_A, class_B = get_class(n, train_imgs, train_labels)

#     # Define the parameters
#     lambda_param = 1
#     nu = 2*len(class_A.vectors) + 2*len(class_B.vectors) + 2
#     epsilon = 1e-6

#     # Run the optimization
#     t0 = time.process_time()
#     _, _, _,  _, _, ite = short_step_method(class_A, class_B, lambda_param, nu, epsilon)
#     tf  = time.process_time()

#     timeCPU.append(tf - t0)
#     nIte.append(ite)

timeCPU = [897.6654179999969, 1441.598113999993, 2075.079472000005, 4410.586334, 5513.837536999999, 5864.554860999997, 6786.646188999999, 6424.781306000004, 7068.133726999993, 20187.240282999992]
nIte = [3059, 3733, 4303, 4820, 5278 ,5699, 6092, 6461, 6811,9715]

print("timeCPU : ", *timeCPU)
print("nIte : ", *nIte)

fig , ax = plt.subplots(1, 2, figsize=(12, 5))
ax = ax.ravel()

ax[0].plot(nImage, timeCPU, "^-", color = "k")
ax[0].set_xlabel("Number of images per Class")
ax[0].set_ylabel("CPU time (s)")
ax[0].set_title("CPU time")
ax[0].set_xticks(np.arange(5, 25, 5))

ax[1].plot(nImage, nIte, ".-")
ax[1].set_xlabel("Number of images per Class")
ax[1].set_ylabel("Number of iterations")
ax[1].set_title("Number of iterations")
ax[1].set_xticks(np.arange(5, 25, 5))

plt.savefig("short_step.pdf")
plt.show()