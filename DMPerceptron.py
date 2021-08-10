import numpy as np 
import random 

# general binary training percepton 
def perceptronTrain(trainingLabels, trainingData, maxIter):

    #No of training objects
    numTrainingObj = len(trainingData)

    #No of features in training (looks at the object at position zero and counts the length)
    numFeatures = len(trainingData[0])

    #intilize the bias term and weights to zero
    W = np.zeros(numFeatures)
    b = 0

    for t in range(maxIter):
        for i in range(numTrainingObj):
            X = trainingData[i]
            # correct label y
            y = trainingLabels[i]

            # compute activation a
            a = np.dot(X,W) + b
    
            if (y*a) <= 0: # if it's a misclassification then do this 
                # update the bias term and the weights
                W = W + (y*X) 
                b = b + y
    return (b,W) 
                     
def perceptronTest(b,W,X): 

    # compute activation a
    a = np.dot(X,W) + b

    return np.sign(a)

# read data into program
def loadData(fname):
 
    labels = []
    features = []
    
    # loop through the file line by line
    with open(fname) as F:
        for line in F:
            # remove the whitespace and then split the data where there is a comma
            p = line.strip().split(',')

            if p[-1] == 'class-1':
                labels.append(1)
            elif p[-1] == 'class-2':
                labels.append(2)
            elif p[-1] == 'class-3':
                labels.append(3)

            # append all the items up until the penultimate item to the features array 
            features.append(np.array(p[:-1], float))

    # return both arrays when the function is called
    return np.array(labels), np.array(features)

# takes the parameters features and data and a given positive or negative class 
# as set by the user to make the list only have 
def makeBinary(trainingLabels, trainingFeatures, positiveCls, negativeCls):
 
    binaryFeatures = []
    binaryLabels = []
    
    # starting index counter at -1 as the position starts 0 in a list/array
    index = -1 

    # loop through the labels 
    for x in trainingLabels:
        # increasing the counter for each itteration to keep track of the index to ensure the labels and features at both at the same index
        index = index + 1
        if x == positiveCls:
            binaryLabels.append(1)
            binaryFeatures.append(np.array(trainingFeatures[index], float))
        elif x == negativeCls:
            binaryLabels.append(-1)
            binaryFeatures.append(np.array(trainingFeatures[index], float))

    # converting list to an array 
    binaryLabelsArr = np.array(binaryLabels) 
    binaryFeaturesArr = np.array(binaryFeatures)

    # shuffle data
    np.random.seed(22)
    permutation = np.random.permutation(len(binaryLabels))
    binaryLabelsArr = binaryLabelsArr[permutation]
    binaryFeaturesArr = binaryFeaturesArr[permutation]

    return binaryLabelsArr, binaryFeaturesArr

# calculate accuracy 
def evaluationReport(classTrue, classPred):
   
    # counts the number of correct predictions
    correctPred = 0 

    for i in range(len(classTrue)):
        if classTrue[i] == classPred[i]:
            correctPred += 1

    accuracy = (correctPred/len(classPred))*100

    # print("Evaluation report")
    return ("%.2f" % accuracy)

 
def binary(positiveCls, negativeCls):
    # loading in data for training and test files
    (trainingLabels, trainingFeatures) = loadData('train.data')
    (testLabels, testFeatures) = loadData('test.data')

    # take the input and make it a binary label array of 1 and -1 depending on the defined class for test and training data
    (binTrainingLabels, binTrainingFeatures) = makeBinary(trainingLabels, trainingFeatures, positiveCls, negativeCls)
    (binTestLabels, binTestFeatures) = makeBinary(testLabels, testFeatures, positiveCls, negativeCls)

    # run the training perceptron with the training data to get the bias and weight
    (b,W) = perceptronTrain(binTrainingLabels, binTrainingFeatures, 20)
    print("Bias term: ", b, "\nWeight vector: ", W) 

    #iterate through the training labels to get the correct labels to compare in the accuracy function
    classTrue = np.array([int(x) for x in binTrainingLabels], dtype=int)
    
    # sub in the b and W into the test perceptron while looping through each feature in the training list to calc the predicted labels
    classPred = np.array([int(perceptronTest(b,W,X)) for X in binTrainingFeatures], dtype=int)
 
    # sub in the array of predicted and true labels to calculate the accuracy of the training data 
    print('train accuracy:',evaluationReport(classTrue, classPred))
   
    # repeat for the test data 
    classTrueTest = np.array([int(x) for x in binTestLabels], dtype=int)
    classPredTest = np.array([int(perceptronTest(b,W,X)) for X in binTestFeatures], dtype=int)
    
    print("test accuracy:",evaluationReport(classTrueTest, classPredTest))


        """Runs binary classification funciton for each specified comparison of classes"""

print('Class 1 and class 2\n')
binary(1, 2)
print('\n')
print('Class 2 and class 3\n')
binary(2, 3)
print('\n')
print('Class 1 and class 3\n')
binary(1, 3)
print('\n\n')
