
import pandas as pd
from methods import *
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
  

col_names = [f"a{i}" for i in range(1, 5)] + ["b"]
iris = pd.read_csv("iris/iris.data", names=col_names)
iris["a0"] = 1


b = pd.get_dummies(iris["b"])["Iris-virginica"].to_numpy(dtype=int)
a = iris[["a0"] + col_names[:-1]].to_numpy()

# a_train, a_test, b_train, b_test = train_test_split(a, b, test_size = .20, random_state = 42)



##----------------------PART I ----------------------##


methods = [sub_gradient_method, proximal_gradient_method, accelerated_proximal_gradient_method]
methods_name = ["Sub Gradient Method", "Proximal Gradient Method", "Accelerated Proximal\nGradient Method"]


k = np.arange(MAXITER)
plt.figure()

for method, name in zip(methods, methods_name):
    func_val = method(a, b, plot = True)
    plt.plot(k, func_val, label = f"{name}")

plt.xlabel("Number of Iteration [k]")
plt.ylabel(r"$min_{i=0,...,k}~~F\left(x_i\right)$")

plt.legend(shadow = True)
plt.grid(True)
plt.show()


##----------------------PART II ----------------------##

MIN = 58.86560070238883 # Empirically found

plt.figure()
func_val = accelerated_proximal_gradient_method(a, b, plot = True)


plt.semilogy(k, func_val - MIN, label = f"Accelerated Proximal\nGradient Method", color = "tab:green")
plt.semilogy(k, 2*(2.25)**2 *2350/(k+1)**2 , "--", label = r"$2L\frac{||x_0 - x^*||^2}{(k+1)^2}$", color = "red")
plt.xlabel("Number of Iteration [k]")
plt.ylabel(r"$min_{i=0,...,k}~~F\left(x_i\right) - F(x_*)$")
plt.legend(shadow = True)
plt.grid(True)
plt.show()

plt.figure()
func_val = proximal_gradient_method(a, b, plot = True)
plt.semilogy(k, func_val - MIN, label = f"Proximal Gradient Method", color = "tab:orange")
plt.semilogy(k, 1/2*(2.25)**2 * 2340/(k+1) , "--", label = r"$L\frac{||x_0 - x^*||^2}{2(k+1)}$", color = "red")
plt.xlabel("Number of Iteration [k]")
plt.ylabel(r"$min_{i=0,...,k}~~F\left(x_i\right) - F(x_*)$")
plt.legend(shadow = True)
plt.grid(True)
plt.show()


plt.figure()
func_val = sub_gradient_method(a, b, plot = True)
plt.semilogy(k, func_val - MIN, label = f"Sub Gradient Method", color = "tab:blue")
plt.semilogy(k, 225*2.4 / np.sqrt(k+1), "--", label = r"$M\frac{||x_0 - x^*||}{(k+1)^{1/2}}$", color = "red")
plt.xlabel("Number of Iteration [k]")
plt.ylabel(r"$min_{i=0,...,k}~~F\left(x_i\right) - F(x_*)$")
plt.legend(shadow = True)
plt.grid(True)
plt.show()







