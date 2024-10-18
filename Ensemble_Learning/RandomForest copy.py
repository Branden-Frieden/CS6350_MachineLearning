import pandas as pd
import math
import numpy as np
import random
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
    


def EntropyPurity(counts):
    totalCounts = sum(counts)
    purity = 0
    for count in counts:
        p_i = count/totalCounts
        if p_i == 0:
            continue
        purity -= p_i * math.log(p_i,2)
    return purity

def MajorityErrorPurity(counts):
    totalCounts = sum(counts)
    if totalCounts == 0:
        return 0
    return (min(counts) / totalCounts)

def GiniIndexPurity(counts):
    totalCounts = sum(counts)
    if totalCounts == 0:
        return 0
    output = 0
    for count in counts:
        p_i = count / totalCounts
        output += p_i ** 2
    return 1 - output

def checkPurity(examples, attributes, labelValues,  purityEquation):
    
    labelCounts = set()
    # add counts for each label type to calculate set purity with
    for label in labelValues:
        labelCounts.add(examples[examples['label'] == label].shape[0])
    if purityEquation == 'entropy':
        purity = EntropyPurity(labelCounts)
    elif purityEquation == 'majorityError':
        purity = MajorityErrorPurity(labelCounts)
    elif purityEquation == 'giniIndex':
        purity = GiniIndexPurity(labelCounts)
    else :
        raise ValueError(f"Invalid purity equation: {purityEquation}  |  Expected 'entropy', 'majorityError', or 'giniIndex'.")
    
    return purity


def findBestSplit(examples, attributes, labelValues, purityEquation):
    bestSplit = None
    bestInformationGain = -1
    
    
    setPurity = checkPurity(examples, attributes, labelValues, purityEquation)

    for attribute in attributes:
        if attribute == 'label':
            continue
        attributeValues = examples[attribute].unique()
        informationGain = setPurity

        for attributeValue in attributeValues:
            subExamples = examples[examples[attribute] == attributeValue]
            newPurity = checkPurity(subExamples, attributes, labelValues, purityEquation)
            informationGain -= (subExamples.shape[0] / examples.shape[0]) * newPurity

        if informationGain > bestInformationGain:
            bestInformationGain = informationGain
            bestSplit = attribute

    return bestSplit
                

def BuildID3Tree(examples, attributes, labelValues, node, purityEquation, maxDepth: int, attributeSubset):
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
        node.setLabel(labelCounts.idxmax())
        return
    
    # find the best attribute to split on
    if attributeSubset < len(attributes.keys()):
        subAttributes = random.sample(list(attributes.keys()), attributeSubset)
    else :
        subAttributes = attributes
    bestAttribute = findBestSplit(examples, subAttributes, labelValues, purityEquation)
    node.setLabel(bestAttribute)
    # print("best attribute to split on is: ", bestAttribute)
    #clear bestAttribute from the new attribute list
    newAttributes = attributes.copy()
    newAttributes.pop(bestAttribute)
    #split the data based on the attribute
    for a in attributes[bestAttribute]:
        newExamples = examples[examples[bestAttribute] == a]
        newNode = Node()
        newNode.setParent(node)
        newNode.setDepth(node.depth + 1)
        newNode.splitOn = a
        node.setChild(newNode)
        BuildID3Tree(newExamples, newAttributes, labelValues,  newNode, purityEquation, maxDepth, attributeSubset)


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
    

def baggedError(examples, classifiers):
    n = len(examples)
    predictions = np.zeros(n)
    for classifier in classifiers:
        preds = np.array([ID3TreeSearch(examples.iloc[i], classifier) for i in range(n)])
        preds = np.where(preds == 'yes', 1, -1)
        predictions += preds

    final_predictions = np.where(predictions > 0, 'yes', 'no')

    return 1 - sum(final_predictions == examples['label']) / len(examples)

def randomForest(examples,test_data, attributes, labelValues, T, subsetSize):
    purityEquation = 'entropy'
    classifiers = []
    train_errors = []
    test_errors = []

    for t in range(T):
        # pick m data points (about half of the total number of examples w/ replacement)
        m = 1000
        subExamples = examples.sample(n=m, replace=True)
        rootNode = Node()
        BuildID3Tree(subExamples, attributes, labelValues, rootNode, purityEquation, len(attributes) - 1, subsetSize)

        classifiers.append(rootNode)
        train_errors.append(baggedError(subExamples, classifiers))
        test_errors.append(baggedError(test_data, classifiers))


    return classifiers, train_errors, test_errors



subsetSizes = [2,4,6]
T = 15
total_train_errors = []
total_test_errors = []

plt.figure(figsize=(12,6))

for subsetSize in subsetSizes:
    forests, train_errors, test_errors = randomForest(train_data, test_data, attributes, labels,  T, subsetSize)
    total_train_errors.append(train_errors)
    total_test_errors.append(test_errors)

    for t in range(len(train_errors)):
        print("subsetSize: ", subsetSize, " t: ", t,  " train_error: ", train_errors[t], "test_error: ", test_errors[t])

    plt.plot(list(range(len(train_errors))), train_errors, marker='o', label=f'train errors for subset size: {subsetSize} ')
    plt.plot(list(range(len(test_errors))), test_errors, marker='o', label=f'test errors for subset size: {subsetSize} ')

plt.title('training and testing errors for different subset sizes and number of trees in a random forest')
plt.xlabel('number of trees')
plt.ylabel('Error')
plt.legend()
plt.grid()
plt.xticks(subsetSizes)  # Ensure subset sizes are shown on x-axis
plt.show()