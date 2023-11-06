from ucimlrepo import fetch_ucirepo 
import pandas as pd
from methods import *
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
  
# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
a = iris.data.features.to_numpy() 
b = pd.get_dummies(iris.data.targets)
b_target = np.array(b["class_Iris-virginica"], dtype = int)

a_train, a_test, b_train, b_test = train_test_split(a, b_target, test_size = .25)

methods = [sub_gradient_method, proximal_gradient_method, accelerated_proximal_gradient_method]
methods_name = ["sub gradient", "proximal gradient", "accelerated proximal gradient"]


# for method, name in zip(methods, methods_name):
#     xopt = method(a_train, b_train)
#     b_pred = classify(xopt[-1], a_test)
#     print(f"Method : {name}")
#     print(f"=> Accuracy : {accuracy_score(b_test, b_pred)}\n")


k = np.arange(MAXITER)
fig, ax = plt.subplots(1, 2)

for method, name in zip(methods, methods_name):
    func_val = method(a, b_target, plot = True)
    ax[0].plot(k, func_val, label = f"{name}")

ax[0].set_xlabel("Number of Iteration")
ax[0].set_ylabel("Objective Function Value")
ax[0].legend(shadow = True)


for method, name in zip(methods, methods_name):
    Xopt = method(a_train, b_train)
    b_pred = [classify(xopt, a_test) for xopt in Xopt]
    acc = [accuracy_score(pred, b_test) for pred in b_pred]
    ax[1].plot(k, acc, label = f"{name}")

ax[1].set_xlabel("Number of Iteration")
ax[1].set_ylabel("Model Accuracy")
ax[1].legend(shadow = True)

plt.show()







