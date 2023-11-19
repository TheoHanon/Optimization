import pandas as pd
from methods import *
import numpy as np
import matplotlib.pyplot as plt 



col_names = ["a1","a2","a3","a4","b"]
iris = pd.read_csv("iris/iris.data", names = col_names)
iris["a0"] = 1

b = pd.get_dummies(iris["b"])
b = np.array(b["Iris-virginica"], dtype = int)
a = iris[["a0","a1", "a2", "a3", "a4"]].to_numpy()

methods = [sub_gradient_method, proximal_gradient_method, accelerated_proximal_gradient_method]
methods_name = ["sub gradient", "proximal gradient", "accelerated proximal gradient"]


def numberOfIteration(epsilon, f_value):
    f_low = min(f_value)
    for k, val in enumerate(f_value):
        if (val - f_low <= epsilon): return k

    return len(f_value)

def compute_p(N_val, Epsilon):
    N_val = N_val[Epsilon < 1]
    Epsilon = Epsilon[Epsilon < 1]
    return - np.mean(np.log(N_val) / np.log(Epsilon))


Epsilon = np.linspace(1e-5, .5, 10000)


for method, name in zip(methods, methods_name):
    func_val = method(a, b, plot = True)
    N_val = np.arange(0, func_val.size)
    plt.plot(N_val, func_val - min(func_val), label = f"{name}")
    print(compute_p(N_val, func_val - min(func_val)))

plt.xlabel("Number Of Iteration")
plt.ylabel(r"$\epsilon$")
plt.show()
