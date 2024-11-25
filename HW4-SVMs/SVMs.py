import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from scipy.optimize import minimize
from sklearn.metrics.pairwise import rbf_kernel


def schedule1(gamma_0, alpha, t):
    return gamma_0 / (1 + gamma_0/alpha * t)

def schedule2(gamma_0, alpha, t):
    return gamma_0 / (1 + t)

def primal_svm(X_train, Y_train, C, gamma_0, schedule, alpha, max_epochs):
    n, d = X_train.shape
    w = np.zeros(d)
    b = 0

    for epoch in range(max_epochs):
        X_train, Y_train = shuffle(X_train, Y_train)

        for i in range(n):
            gamma = schedule(gamma_0, alpha, i+1)

            if Y_train[i] * (np.dot(w, X_train[i]) + b) >= 1:
                w -= gamma * w
            else:
                w -= gamma * (w-C*Y_train[i]*X_train[i])
                b += gamma * C * Y_train[i]
    return w, b



def dual_svm(X, y, C):
    n = len(y)

    alpha_0 = np.zeros(n)

    alpha = minimize()




train_data = pd.read_csv("train.csv", header=None)
test_data = pd.read_csv("test.csv", header=None)
print(train_data)
X_train = train_data.iloc[:, :-1].values
Y_train = train_data.iloc[:, -1].values
Y_train = np.where(Y_train == 0, -1, 1)

X_test = test_data.iloc[:, :-1].values
Y_test = test_data.iloc[:, -1].values
Y_test = np.where(Y_test == 0, -1, 1)

Cs = [100/873, 500/873, 700/873]

max_epochs = 100
gamma_0 = .03
alpha = 1


for C in Cs:
    w, b = primal_svm(X_train, Y_train, C, gamma_0, schedule1, alpha, max_epochs)
    train_acc = np.mean(np.sign(np.dot(X_train, w) + b) == Y_train)
    test_acc = np.mean(np.sign(np.dot(X_test, w) + b) == Y_test)
    
    print(f"for schedule 1, with C = {C}, train_acc = {train_acc}, test_acc = {test_acc}")
    print(f"w = {w}, b = {b}\n")

print("\n\n")
for C in Cs:
    w, b = primal_svm(X_train, Y_train, C, gamma_0, schedule2, alpha, max_epochs)
    train_acc = np.mean(np.sign(np.dot(X_train, w) + b) == Y_train)
    test_acc = np.mean(np.sign(np.dot(X_test, w) + b) == Y_test)
    
    print(f"for schedule 2, with C = {C}, train_acc = {train_acc}, test_acc = {test_acc}")
    print(f"w = {w}, b = {b}\n")

