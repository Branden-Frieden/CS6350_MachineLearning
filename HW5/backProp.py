import numpy as np
import torch
import torch.nn as nn
import math
import pandas as pd
from matplotlib import pyplot as plt

def sig(x):
    return 1/ (1 + np.exp(-x))

def forwardPass(X, W1, W2, W3):
    #first layer
    # print(X.shape)
    # print(W1.shape)
    Z1 = np.dot(X, W1)
    # print(Z1)
    Z1sig = np.insert(sig(Z1), 0, 1)
    Z2 = np.dot(Z1sig, W2)
    Z2sig = np.insert(sig(Z2), 0 ,1)
    y = np.dot(Z2sig, W3)

    # print(Z1)
    # print(Z1sig)
    # print(Z2)
    # print(Z2sig)
    # print(y)
    return y, Z1, Z1sig, Z2, Z2sig                       



def backwardPass(X, y_correct, W1, W2, W3):
    

    # Loss
    y, Z1, Z1sig, Z2, Z2sig = forwardPass(X, W1, W2, W3)
    loss = 1.0/2.0*math.pow((y-y_correct), 2)
    # print("loss = ", loss)

    # backward pass
    dLdy = y-y_correct
    # print("dLdy: ",dLdy)

    dLdW3 = np.outer(Z2sig.T, dLdy)
    # print("dLdW3: ", dLdW3)

    dLdZ2 = np.dot(dLdy, W3.T)[1:] * sig(Z2)*(1-sig(Z2))
    # print("dLdZ2: ", dLdZ2)

    dLdW2 = np.outer(Z1sig, dLdZ2)
    # print("dLdW2: ", dLdW2)

    dLdZ1 = np.dot(dLdZ2, W2.T)[1:] * sig(Z1)*(1-sig(Z1))
    # print("dLdZ1: ", dLdZ1)
    
    dLdW1 = np.outer(X, dLdZ1.T)
    # print("dLdW1: ", dLdW1)
    return loss, dLdW3, dLdW2, dLdW1

def update_weights(W1, W2, W3, dW1, dW2, dW3, initial_lr, schedule_t):
    lr = initial_lr / (1 + initial_lr/10 * schedule_t)
    W1 -= dW1 * lr
    W2 -= dW2 * lr
    W3 -= dW3 * lr
    return

# check against manual calculation
# W1 = np.array([-1, 1, -2, 2, -3, 3]).reshape(3, 2)
# W2 = np.array([-1, 1, -2, 2, -3, 3]).reshape(3, 2)
# W3 = np.array([-1, 2, -1.5])
# W1d, W2d, W3d = backwardPass(X, 1.0, W1, W2, W3)

initial_lr = .001
epochs = 100


df = pd.read_csv("train.csv", header=None)
df.insert(0, 'bias', 1)

testDF = pd.read_csv("test.csv", header=None)
testDF.insert(0, 'bias', 1)
print(testDF.shape)

for layerWidths in [5, 10, 25, 50, 100]:
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    W1 = np.zeros((X.shape[1], layerWidths))
    W2 = np.zeros((layerWidths+1, layerWidths))
    W3 = np.zeros((layerWidths+1,1))
    initial_lr = .01


    losses = []
    for epoch in range(epochs):
        #shuffle data
        df = df.sample(frac=1).reset_index(drop=True)
        avgLoss = 0
        for i, X in df.iterrows():
            #run forward and backward pass
            loss, dLdW3, dLdW2, dLdW1 = backwardPass(X.values[:-1], int(X.values[-1]), W1, W2, W3)
            avgLoss += loss
            update_weights(W1, W2, W3, dLdW1, dLdW2, dLdW3, initial_lr, epoch)
        losses.append(avgLoss/df.shape[0])
    print(f"train error for {layerWidths} widths: {losses[-1]}")
    plt.plot(losses)

    #testing
    error = 0
    for i, X in testDF.iterrows():
        testPred, _, _, _, _ = forwardPass(X[:-1], W1, W2, W3)
        error += math.pow(testPred - X.iloc[-1], 2)
    print(f"test error for {layerWidths} widths: {error/testDF.shape[0]}")
# plt.show()