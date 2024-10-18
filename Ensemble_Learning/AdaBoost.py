import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

treatUnkownAsAttribute = True

attributes = {}
labels = {"yes", "no"}
columnLabels = []

with open( 'bank/data-desc.txt' , 'r') as f:
    for line in f:
        words = line.strip().split(" ")
        if words[0].isdigit() and words[1] == '-':
            columnLabels.append(words[2].replace(':', ''))

            if "(binary:" in words:
                attributes[words[2].replace(':', '').replace('\'','')] = ["yes", "no"]




            categoricalWords = []
            insideCategorical = False
            categorical = False
            for word in words:
                # Check if the word indicates the start of a (categorical: block
                if "(categorical:" in word and not insideCategorical:
                    insideCategorical = True
                    categorical = True
                    continue  # Skip adding the word itself
                
                # If we're inside the categorical block, add words to the list
                if insideCategorical:
                    if "\",\"" in word:
                        categoryWords = word.replace("\"", "").replace(")","").split(",")
                        for  categoryWord in categoryWords:
                            if ")" in categoryWord or ";" in categoryWord:  
                                categoricalWords.append(categoryWord.rstrip(';').rstrip("\"").replace("\"", "").replace(")","").replace(",",""))
                            else: 
                                categoricalWords.append(categoryWord)
                        break


                    if ")" in word or ";" in word:  
                        insideCategorical = False
                        categoricalWords.append(word.rstrip(';').rstrip("\"").replace("\"", "").replace(")","").replace(",",""))
                    else:
                        categoricalWords.append(word.rstrip(';').rstrip("\"").replace("\"", "").replace(")","").replace(",",""))
            
            if categorical:
                attributes[words[2].replace(':', '').replace('\'','')] = categoricalWords


# load training data
train_data = pd.read_csv('bank/train.csv', header=None)
train_data.columns = columnLabels
train_data.rename(columns={train_data.columns[-1]: 'label'}, inplace=True)

attributes["age"] = ["yes", "no"]
attributes["job"] = ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services"]
attributes["marital"] = ["married","divorced","single"]
attributes["education"] = ["unknown","secondary","primary","tertiary"]
attributes["default"] = ["yes","no"]
attributes["balance"] = ["yes","no"]
attributes["housing"] = ["yes","no"]
attributes["loan"]  = ["yes","no"]
attributes["contact"]   = [ "unknown","telephone","cellular"]
attributes["day"] = ["yes","no"]
attributes["month"] = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sept", "oct", "nov", "dec"]
attributes["duration"] = ["yes", "no"]
attributes["campaign"] = ["yes", "no"]
attributes["pdays"] = ["yes", "no"]
attributes["previous"] = ["yes", "no"]
attributes["poutcome"] = ["unknown","other","failure","success"]
attributes["label"] = attributes["y"]
attributes.pop("y")
columnLabels.pop()
columnLabels.append("label")



test_data = pd.read_csv('bank/test.csv', header=None)
test_data.columns = columnLabels
test_data.rename(columns={test_data.columns[-1]: 'label'}, inplace=True)



# print to check accuracy
print(labels,"\n")
print(attributes,"\n")
print(columnLabels,"\n")



for column in train_data.columns:
    if train_data[column].dtype == np.int64:
        trainSeries = train_data[column]
        testSeries = test_data[column]

        trainMedValue = trainSeries.median()
        testMedValue = testSeries.median()

        for j in range(len(trainSeries)):
            if train_data[column][j] > trainMedValue:
                train_data.loc[j, column] = "yes"
            else:
                train_data.loc[j, column] = "no"

        for j in range(len(testSeries)):
            if test_data[column][j] > testMedValue:
                test_data.loc[j, column] = "yes"
            else:
                test_data.loc[j, column] = "no"

        
        attributes[column] = ["yes", "no"]

if not treatUnkownAsAttribute:
    for column in train_data.columns:
        if train_data[column].mode()[0] == "unknown":
            columnMode = train_data[column].value_counts().index[1]
        else:
            columnMode = train_data[column].mode()[0]
        
        train_data[column] = train_data[column].replace("unknown", columnMode)

print(train_data)    



class Node():
    # initialize with no parent (root node)
    def __init__(self):
        self.children = []
        self.parent = None
        self.label = None
        self.depth = 0
        self.splitOn = None

    def setParent(self, parent):
        self.parent = parent

    def setChild(self, child):
        self.children.append(child)

    def setLabel(self, label):
        self.label = label

    def setDepth(self, depth):
        self.depth = depth
    


def EntropyPurity(counts, weight):
    totalCounts = sum(counts)
    purity = 0
    if weight == 0:
        return 0
    for count in counts:
        p_i = count/weight
        if p_i == 0:
            continue
        purity -= p_i * math.log(p_i,2)
    return purity

def MajorityErrorPurity(counts, weight):
    totalCounts = sum(counts)
    if weight == 0:
        return 0
    return (min(counts) / weight)

def GiniIndexPurity(counts, weight):
    totalCounts = sum(counts)
    if weight == 0:
        return 0
    output = 0
    for count in counts:
        p_i = count / weight
        output += p_i ** 2
    return 1 - output

def checkPurity(examples, weights, attributes, labelValues,  purityEquation):
    labelCounts = set()
    totalWeight = sum(weights)
    # add counts for each label type to calculate set purity with
    for label in labelValues:
        labelCounts.add(sum(weights[examples['label'] == label]))
    if purityEquation == 'entropy':
        purity = EntropyPurity(labelCounts, totalWeight)
    elif purityEquation == 'majorityError':
        purity = MajorityErrorPurity(labelCounts, totalWeight)
    elif purityEquation == 'giniIndex':
        purity = GiniIndexPurity(labelCounts, totalWeight)
    else :
        raise ValueError(f"Invalid purity equation: {purityEquation}  |  Expected 'entropy', 'majorityError', or 'giniIndex'.")
    
    return purity


def findBestSplit(examples, attributes, labelValues, purityEquation, weights):
    bestSplit = None
    bestInformationGain = -1
    
    
    setPurity = checkPurity(examples, weights, attributes, labelValues, purityEquation)

    for attribute in attributes:
        if attribute == 'label':
            continue
        attributeValues = examples[attribute].unique()
        informationGain = setPurity

        for attributeValue in attributeValues:
            subExamples = examples[examples[attribute] == attributeValue]
            subWeights = weights[examples[attribute] == attributeValue]
            newPurity = checkPurity(subExamples, subWeights, attributes, labelValues, purityEquation)
            informationGain -= (subExamples.shape[0] / examples.shape[0]) * newPurity

        if informationGain > bestInformationGain:
            bestInformationGain = informationGain
            bestSplit = attribute

    return bestSplit
                

def BuildID3Tree(examples, attributes, labelValues, node, purityEquation, maxDepth: int, weights):
    # check if leaf node, if yes, set node label and return
    if examples.empty:
        node.label = "empty"
        return

    if examples["label"].nunique() == 1:
        # print("making leaf node")
        node.label = examples['label'].iloc[0]
        return 0
    
    
    if node.depth >= maxDepth:
        # print("max depth hit, making leaf node")
        labelCounts = examples['label'].value_counts()

        labelWeights = {}
        for i in range(len(examples)):
            label = examples['label'].iloc[i]
            if label not in labelWeights:
                labelWeights[label] = weights[i]
            else:
                labelWeights[label] += weights[i]
            
        node.setLabel(max(labelWeights, key=labelWeights.get))
        return
    
    # find the best attribute to split on
    bestAttribute = findBestSplit(examples, attributes, labelValues, purityEquation, weights)
    node.setLabel(bestAttribute)
    # print("best attribute to split on is: ", bestAttribute)
    #clear bestAttribute from the new attribute list
    newAttributes = attributes.copy()
    newAttributes.pop(bestAttribute)
    #split the data based on the attribute
    for a in attributes[bestAttribute]:
        newExamples = examples[examples[bestAttribute] == a]
        newWeights = weights[examples[bestAttribute] == a]
        newNode = Node()
        newNode.setParent(node)
        newNode.setDepth(node.depth + 1)
        newNode.splitOn = a
        node.setChild(newNode)
        BuildID3Tree(newExamples, newAttributes, labelValues,  newNode, purityEquation, maxDepth, newWeights)


def ID3TreeSearch(example, node):
    if not node.children:
        return node.label
    
    splitLabel = node.label
    attributeValue = example[splitLabel]

    for child in node.children:
        if child.splitOn == attributeValue:
            return ID3TreeSearch(example, child)
    # print(splitLabel, attributeValue)
    return -1
    


def AdaBoost(train_data, attributes, labels, purityEquation, T):
    m = train_data.shape[0]
    weights = np.ones(m) / m
    maxDepth = 2

    classifiers = []
    alphas = []
    

    for t in range(T):
        rootNode = Node()
        BuildID3Tree(train_data, attributes, labels, rootNode, purityEquation, maxDepth, weights)

        preds = []
        errors = []

        for i in range(m):
            datum = train_data.iloc[i]
            prediction = ID3TreeSearch(datum, rootNode)
            preds.append(prediction)
            errors.append((prediction != datum["label"]) * weights[i])
        
        train_error = sum(errors)

        if train_error <= 0:
            classifiers.append(rootNode)
            alphas.append(1)
            break
        
        print("error: ", train_error)
        alpha = 1/2 * math.log((1 - train_error)/train_error)
        
        print("alpha: ", alpha)
        for i in range(m):
            if errors[i] > 0:
                weights[i] *= math.exp(alpha)
            else:
                weights[i] *= math.exp(-alpha)
        
        weights /= sum(weights)

        classifiers.append(rootNode)
        alphas.append(alpha)

    return classifiers, alphas



def evaluateAdaBoost(classifiers, alphas, examples):
    N = examples.shape[0]
    final_predictions = np.zeros(N)

    for classifier, alpha in zip(classifiers, alphas):
        predictions = np.array([ID3TreeSearch(examples.iloc[i], classifier) for i in range(N)])
        predictions = np.where(predictions == 'yes', 1, -1)  # Convert labels to numerical
        final_predictions += alpha * predictions

    final_predictions = np.where(final_predictions > 0, 'yes', 'no')  # Final classification
    error = sum(final_predictions != examples['label']) / N
    return error




def evaluateStumps(classifiers, examples):
    N = examples.shape[0]
    errors = []

    for classifier in classifiers:
        predictions = np.array([ID3TreeSearch(examples.iloc[i], classifier) for i in range(N)])
        error = sum(predictions != examples['label']) / N
        errors.append(error)
    return errors



T = 50
classifiers, alphas = AdaBoost(train_data, attributes, labels,'entropy', T)

train_accuracies = []
test_accuracies = []
for t in range(T):
    train_accuracies.append(evaluateAdaBoost(classifiers[:t], alphas[:t], train_data))
    test_accuracies.append(evaluateAdaBoost(classifiers[:t], alphas[:t], test_data))
    
print(train_accuracies,"\n", test_accuracies)



trainStumpAccuracies = evaluateStumps(classifiers,train_data)
testStumpAccuracies = evaluateStumps(classifiers, test_data)

print(trainStumpAccuracies)
print(testStumpAccuracies)


iterations = list(range(T))


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.plot(iterations, train_accuracies, label='Train error', marker='o')
ax1.plot(iterations, test_accuracies, label='Test error', marker='o')
ax1.set_title('Train and Test error over Iterations')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('error')
ax1.legend()
ax1.grid(True)

ax2.plot(iterations[:len(trainStumpAccuracies)], trainStumpAccuracies, label='Train error', marker='o')
ax2.plot(iterations[:len(testStumpAccuracies)], testStumpAccuracies, label='Test error', marker='o')
ax2.set_title('Train and Test error of Each Iteration\'s Stump')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('error')
ax2.legend()
ax2.grid(True)

# Display the two plots side by side
plt.tight_layout()
plt.show()




