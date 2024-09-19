import pandas as pd
import math

attributes = {}
labels = set()
columnLabels = set()
labelRead = False
attributeRead = False
columnsRead = False

with open( 'car/data-desc.txt' , 'r') as f:
    for line in f:
        if "label" in line:
            labelRead = True
        if "attributes" in line:
            attributeRead = True
        if "columns" in line:
            columnsRead = True

        # grab the labels of the data
        if labelRead and not attributeRead and not columnsRead:
            templabels = set(line.strip().rstrip('.').split(", "))
            if len(templabels) > 1:
                labelRead = False
                labels = templabels


        # grab catagorical attributes
        if attributeRead and not columnsRead:
            if ':' in line:
                values = set()
                a, v = line.strip().rstrip('.').split(':')
                for value in v.strip().split(',') :
                    values.add(value.strip())
                attributes[a] = values    
        
        # grab column Labels
        if columnsRead :
            columnLabels = [value.strip() for value in line.strip().rstrip('.').split(',')]
        




# print to check accuracy
print(labels,"\n")
print(attributes,"\n")
print(columnLabels,"\n")

# load training data
train_data = pd.read_csv('car/train.csv', header=None)
train_data.columns = columnLabels
print(train_data)

test_data = pd.read_csv('car/test.csv', header=None)
test_data.columns = columnLabels

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
                


def BuildID3Tree(examples, attributes, labelValues, node, purityEquation, maxDepth: int):
    # check if leaf node, if yes, set node label and return
    if examples.empty:
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
    bestAttribute = findBestSplit(examples, attributes, labelValues, purityEquation)
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
        BuildID3Tree(newExamples, newAttributes, labelValues,  newNode, purityEquation, maxDepth)


def ID3TreeSearch(example, node):
    if not node.children:
        return node.label
    
    splitLabel = node.label
    attributeValue = example[splitLabel]
    for child in node.children:
        if child.splitOn == attributeValue:

            return ID3TreeSearch(example, child)
    


purityEquations = ['entropy', 'majorityError', 'giniIndex']
maxMaxDepth = 6

for purityEquation in purityEquations:
    results = []
    for maxDepth in range(maxMaxDepth):
        maxDepth += 1
        rootNode = Node()
        BuildID3Tree(train_data, attributes, labels, rootNode, purityEquation, maxDepth)


        trainCorrect = 0
        for i in range(train_data.shape[0]):
            # print(ID3TreeSearch(train_data.iloc[i], rootNode))
            if ID3TreeSearch(train_data.iloc[i], rootNode) == train_data["label"][i]:
                trainCorrect += 1

        testCorrect = 0
        for i in range(test_data.shape[0]):
            if ID3TreeSearch(test_data.iloc[i], rootNode) == test_data["label"][i]:
                testCorrect += 1

        trainAccuracy = trainCorrect / train_data.shape[0]
        testAccuracy = testCorrect / test_data.shape[0]

        results.append([maxDepth, trainAccuracy, testAccuracy])
    df = pd.DataFrame(results, columns=["Max Depth", "Training Accuracy", "Testing Accuracy"])
    print("for purity equation: ", purityEquation,"\n", df, "\n\n")

print("\n\n 1.c)from the tables above, we can see that the training accuracy increases until it gets 100\% correct while the training accuracy stops and begins to drop with increasing max depth.")
