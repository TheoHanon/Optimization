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

def apply_pca(train_data, test_data, n_components):
    pca = PCA(n_components=n_components)
    train_pca = pca.fit_transform(train_data)
    test_pca = pca.transform(test_data)
    return train_pca, test_pca


def import_data(use_pca = False, n_pca = 5):

    data_path = "./"

    # Load data
    train_data = np.loadtxt(data_path + "mnist_train.csv", delimiter=",")
    test_data = np.loadtxt(data_path + "mnist_test.csv", delimiter=",")

    # Normalize
    fac = 0.99 / 255
    train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
    test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01

    train_labels = np.asfarray(train_data[:, :1])
    test_labels = np.asfarray(test_data[:, :1])

    # Function to apply PCA


    # Apply PCA if chosen
    if use_pca:  # 'use_pca' should be a boolean variable set by the user
        train_imgs, test_imgs = apply_pca(train_imgs, test_imgs, n_pca)

    return train_imgs, test_imgs, train_labels, test_labels


def get_class(Nimage, imgs, labels):
    vector_A = imgs[np.where(labels == 0)[0]]
    vector_B = np.concatenate([imgs[np.where(labels == i)[0]][:(Nimage)] for i in range(1, 10)])
    vector_A = vector_A[:(Nimage*9)]

    class_A = A(vector_A)
    class_B = B(vector_B)

    return class_A, class_B


def classify(x, h, c):
    return 0 if np.dot(h, x,) + c < -1 else 1
