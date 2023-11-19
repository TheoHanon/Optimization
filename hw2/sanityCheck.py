from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from methods import *



col_names = ["a1","a2","a3","a4","b"]
iris = pd.read_csv("iris/iris.data", names = col_names)
iris["a0"] = 1

b = pd.get_dummies(iris["b"])
b = np.array(b["Iris-virginica"], dtype = int)
a = iris[["a0","a1", "a2", "a3", "a4"]].to_numpy()

a_train, a_test, b_train, b_test = train_test_split(a, b, test_size = .20, random_state = 42)

sklearn_model = LogisticRegression(penalty='l1', solver='liblinear', C=1/LAMBDA, max_iter=500)

# Fit the model on the training data
sklearn_model.fit(a_train, b_train)

# Predict labels for the test set
b_pred_sklearn = sklearn_model.predict(a_test)

# Calculate the accuracy for the scikit-learn model
accuracy_sklearn = accuracy_score(b_test, b_pred_sklearn)


xopt = accelerated_proximal_gradient_method(a_train, b_train)
b_pred = classify(xopt[-1], a_test)

print(f"My method Accuracy : {accuracy_score(b_test, b_pred)}")
print(f"Scikit-learn Logistic Regression Accuracy: {accuracy_sklearn}")
