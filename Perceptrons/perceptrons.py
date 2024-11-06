import numpy as np
import pandas as pd

trainData = pd.read_csv("bank-note/train.csv", header=None)
testData = pd.read_csv("bank-note/test.csv", header=None)

print(trainData)

trainData, trainLabels = trainData.iloc[:, :-1].values, trainData.iloc[:, -1].values

testData, testLabels = testData.iloc[:, :-1].values, testData.iloc[:, -1].values


def perceptronTrain(data, labels):
    print(data.shape)
    print(labels.shape)
    T = 10
    w = np.zeros(data.shape[1])
    for e in range(T):
        for i in range(data.shape[0]):
            if np.sign(np.dot(w, data[i])) != labels[i]:
                w += labels[i] * data[i]
    return w

def votedPerceptronTrain(data, labels):
    print(data.shape)
    print(labels.shape)
    T = 10
    w = np.zeros(data.shape[1])
    weights = []
    count = 1
    for e in range(T):
        for i in range(data.shape[0]):
            if np.sign(np.dot(w, data[i])) != labels[i]:
                weights.append((w.copy(), count))

                w += labels[i] * data[i]
                count = 0
            count += 1
    weights.append((w.copy(), count))
    return weights

def averagedPerceptronTrain(data, labels):
    print(data.shape)
    print(labels.shape)
    T = 10
    w = np.zeros(data.shape[1])
    a = np.zeros(w.shape[0])
    for e in range(T):
        for i in range(data.shape[0]):
            if np.sign(np.dot(w, data[i])) != labels[i]:
                w += labels[i] * data[i]
            a += w

    return a / (T * data.shape[0])




def perceptronTest(data, labels, weights):
    return np.mean(np.sign(np.dot(data, weights)) != labels)





# normal perceptron
w = perceptronTrain(trainData, trainLabels)
error = perceptronTest(testData, testLabels, w)

print(w)
print(error)

# voted perceptron
weights = votedPerceptronTrain(trainData, trainLabels)

top_weights = sorted(weights, key=lambda x: x[1], reverse=True)[:5]

total_error = 0
for weight, count in weights:
    error = perceptronTest(testData, testLabels, weight)
    total_error += error

average_error = total_error / len(weights)


for weight, count in top_weights:
    print("Weight:", weight, "Count:", count)

print("Average Error:", average_error)

# averaged perceptron
w = averagedPerceptronTrain(trainData, trainLabels)
error = perceptronTest(testData, testLabels, w)

print(w)
print(error)

