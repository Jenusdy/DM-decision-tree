from math import log
import operator

def createDataSet():
    dataset = [
                [0,1,1,'yes'],
                [0,1,0,'no'],
                [1,0,1,'no'],
                [1,1,1,'no'],
                [0,1,0,'no'],
                [0,0,1,'no'],
                [1,0,1,'no'],
                [1,1,1,'no'],
    ]
    labels = ['cartoon', 'winter','more than 1 person']
    return dataset, labels


def createTree(dataset, labels):

    classList = [example[-1] for example in dataset]

    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataset[0])==1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataset)
    bestFeatLabel = labels[bestFeat]

    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataset, bestFeat, value), subLabels)
    return myTree

def calcShannonEnt(dataset):
    numEntries = len(dataset)
    lbabelCounts = {}
    for featVec in dataset:
        currentLabel = featVec[-1]
        if currentLabel not in lbabelCounts.keys() : lbabelCounts[currentLabel] = 0
        lbabelCounts[currentLabel] += 1
    shannonEnt = 0
    for key in lbabelCounts:
        prob = float(lbabelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0;
    bestFeature = -1
    for i in range(numFeatures):  # iterate over all the features
        featList = [example[i] for example in dataSet]  # create a list of all the examples of this feature
        uniqueVals = set(featList)  # get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)


        infoGain = baseEntropy - newEntropy  # calculate the info gain; ie reduction in entropy
        """
        print("feature : " + str(i))
        print("baseEntropy : "+str(baseEntropy))
        print("newEntropy : " + str(newEntropy))
        print("infoGain : " + str(infoGain))
        """
        if (infoGain > bestInfoGain):  # compare this to the best gain so far
            bestInfoGain = infoGain  # if better than current best, set to best
            bestFeature = i
    return bestFeature  # returns an integer


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]

    if isinstance(valueOfFeat, dict):
        classLabel =  classify(valueOfFeat, featLabels, testVec)
    else :
        classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


myDat, labels = createDataSet()

myTree = createTree(myDat, labels)
print(myTree)

print("Thanks, now I can recognize winter family photo, give me any photo")

answer = classify(myTree, ['cartoon','winter','more than 1 person'],[0,1,1])
print("Hi, the answer is "+ answer +" it is winter family photo")

answer = classify(myTree, ['cartoon','winter','more than 1 person'],[1,1,1])
print("Hi, the answer is "+ answer +" it is not winter family photo")
