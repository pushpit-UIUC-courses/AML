import pandas as pd
import numpy as np
from scipy.stats import norm
import random
import math
raw_data = pd.read_csv('/Users/psaxena21/Documents/Mine/AML/pima-indians-diabetes.csv', header=None)
d = raw_data.values


class NaiveBayes:
    def __init__(self, ignoreMissingVal):
        self.ignoreMissingVal = ignoreMissingVal

    def testTrainSplit(self, data, ratio):
        localCopy = list(data)
        testSize = int(len(data) * ratio)
        testData = []
        while len(testData) < testSize:
            testData.append(localCopy.pop(random.randrange(len(localCopy))))
        testNPArr  = np.array(testData)
        trainNPArr = np.array(localCopy)
        return trainNPArr[:, :8], trainNPArr[:, 8].astype(int), testNPArr[:, :8], testNPArr[:, 8].astype(int)
            
    def fit(self, X, Y):
        self.normDF = {}
        self.priors = {}
        categories = set(Y)
        if self.ignoreMissingVal:
            X[X == 0] = np.nan
        for c in categories:
            XForC = X[Y == c]
            self.normDF[c] = {
                'mean' : np.nanmean(XForC, axis=0),
                'var': np.nanvar(XForC, axis=0)
            }
            self.priors[c] = 1.0 * len(Y[Y == c])/ len(Y)
            
    def __calculateProbability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
        return (1.0 / (math.sqrt(2*math.pi) * stdev)) * exponent
    
    def __calculateLogNormPdf(self, X, mean, stddev):
        local_X = X[X != 0] if self.ignoreMissingVal else X
        local_mean = mean[X != 0] if self.ignoreMissingVal else mean
        local_stddev = stddev[X != 0] if self.ignoreMissingVal else stddev
        return np.log(np.array([self.__calculateProbability(local_X[i], local_mean[i], local_stddev[i]) for i in range(len(local_X))]))
    
    def predict(self, X):
        P = {}
        for c, g in self.normDF.items():
            # print "c:", c
            mean, var = g['mean'], g['var']
            classConditionalProb = 0
            classConditionalProb = np.nansum(np.log(norm.pdf(X, mean, np.sqrt(var))))
            P[c] = classConditionalProb + np.log(self.priors[c])  

            bestCategory, bestProb = None, float("-inf")
        for category, probability in P.items():
            if bestCategory is None or probability > bestProb:
                bestProb = probability
                bestCategory = category
        return bestCategory

def runClassifier(data, ignoreMissingVal=False):
    accuracy = 0
    for i in range(10):
        nb = NaiveBayes(ignoreMissingVal)
        np.random.shuffle(data)
        X_Train, Y_Train, X_Test, Y_Test = nb.testTrainSplit(data, 0.20)
        nb.fit(X_Train, Y_Train)
        correct = 0
        for i in range(len(Y_Test)):
            Y_Pred = nb.predict(X_Test[i])
            if Y_Test[i] == Y_Pred:
                correct += 1
        accuracy += (correct/float(len(Y_Test)) * 100.0)
    return accuracy/10

acc = runClassifier(d)
print("Average accuracy over 10 test-train splits and without ignoring missing values is %f" % (acc))

acc = runClassifier(d, True)
print("Average accuracy over 10 test-train splits and ignoring missing values is %f" % (acc))
