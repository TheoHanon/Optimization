import numpy as np
import matplotlib.pyplot as plt
from HW3_B2 import*
from HW3_B4 import*
from sklearn.metrics import accuracy_score
# from sklea

NIMAGE = 50

image_size = 28 # width and length
no_of_different_labels = 10
image_pixels = image_size * image_size
data_path = "./"
train_data = np.loadtxt(data_path + "mnist_train.csv", 
                        delimiter=",")
test_data = np.loadtxt(data_path + "mnist_test.csv", 
                       delimiter=",") 

fac = 0.99 / 255
train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01

train_labels = np.asfarray(train_data[:, :1])
test_labels = np.asfarray(test_data[:, :1])

vector_A = train_imgs[np.where(train_labels == 0)[0]]
vector_B = train_imgs[np.where(train_labels != 0)[0]]

#Keep only NIMAGE images
vector_A = vector_A[:NIMAGE]
vector_B = vector_B[:NIMAGE]



# Initialize class instances
class_A = A(vector_A)
class_B = B(vector_B)

# Set the lambda parameter for the optimization problem
lambda_param = 1
nu = 2 * len(class_A.vectors) + 2 * len(class_B.vectors) + 2
epsilon = 1e-6


# h0, c0, s0, t0, v0  = init(class_A, class_B)
h, c, s, t, v       = long_step_method(class_A, class_B, lambda_param, nu, epsilon)


l = lambda x: np.dot(h, x) + c

def classify(x):
    return 0 if l(x) <= -1 else 1


y_pred = np.array([classify(im) for im in test_imgs])
test_labels = (test_labels > 0)*1.0
print("Accuracy: ", accuracy_score(test_labels, y_pred))



