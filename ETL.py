__author__ = 'Sriganesh'

import scipy.io
from sklearn import svm
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
import random


def loadData(filename):
    data = scipy.io.loadmat(filename)
    labels, index = getLabels(data['i'])
    trials = []
    k = 0
    for y in data['d']:
        if k in index:
            k += 1
            continue
        i = 0
        while i < 32:
            trials.append(y[0][i])
            i += 1
        k += 1
    return trials, labels

def getLabels(info):
    labels = []
    indexes = []
    i = 0
    for x in info[0]:
        if x[0][0] == 0 or x[0][0] == 1:
            indexes.append(i)
            i += 1
            continue
        if x[len(x)-1][0] == 'P':
            for k in range(0,16):
                labels.append('P')
            for k in range(0,16):
                labels.append('S')
        else:
            for k in range(0,16):
                labels.append('S')
            for k in range(0,16):
                labels.append('P')
        i += 1
    return labels, indexes

def cross_validate(traindata, trainlabels):
    n = 16
    limit = len(traindata)/n
    i = 0
    k_val = [1, 3, 7, 9, 15]
    linearSVM, Gauss, knn = [],[],[]
    while i < n:
        start = i * limit
        end = (i+1) * limit
        if start == 0:
            train = traindata[end:]
            tlabel = trainlabels[end:]
        if end == len(traindata):
            train = traindata[:start]
            tlabel = trainlabels[:start]
        if start != 0 and end != len(traindata):
            train = traindata[:start]
            train.extend(traindata[end:])
            tlabel = trainlabels[:start]
            tlabel.extend(trainlabels[end:])
        test = data[start:end]
        label = trainlabels[start:end]
        L = []
        clf = svm.SVC(C=5000, kernel='linear')
        L.append(fitAndPredict(clf,train,tlabel,test,label))
        linearSVM.append(L)
        clf = GaussianNB()
        Gauss.append(fitAndPredict(clf,train,tlabel,test,label))
        L = []
        for j in k_val:
            if j < len(train[0]):
                clf = KNeighborsClassifier(n_neighbors=j)
                L.append(fitAndPredict(clf,train,tlabel,test,label))
        if len(L) > 0:
            knn.append(L)
        i += 1
    line1, = plt.plot(linearSVM, label= "SVM with C=5000")
    line2, = plt.plot(Gauss, label= "Gaussian")
    X = np.asarray(knn)
    line3, leg3 = [line1, line2], ["SVM with C=5000","Gaussian"]
    for i in range(len(knn[0])):
        line, = plt.plot(X[:,i], label= str(k_val[i])+"nn")
        line3.append(line)
        leg3.append(str(k_val[i])+"nn")
    plt.legend(line3,leg3)
    plt.show()

def fitAndPredict(clf, traindata, trainlabels, testdata,testlabels):
    clf.fit(traindata, trainlabels)
    i , correct = 0, 0
    for x in testdata:
        if testlabels[i] == clf.predict(x):
            correct += 1
        i += 1
    return float(correct)/float(len(testlabels))

def shuffle(data, labels):
    k = len(data) - 144
    print k
    x = [i for i in range(len(data))]
    print len(x)
    random.shuffle(x)
    traindata = [data[i] for i in x[:k]]
    trainlabels = [labels[i] for i in x[:k]]
    testdata = [data[i] for i in x[k:]]
    testlabels = [labels[i] for i in x[k:]]
    return (traindata,trainlabels,testdata,testlabels)

def trainAndtest(traindata, trainlabels, testdata, testlabels):
    linearSVM, Gaussian, knn = [], [], []
    clf = svm.SVC(C=5000,kernel="linear")
    linearSVM.append(fitAndPredict(clf, traindata, trainlabels,testdata, testlabels))
    Gauss = GaussianNB()
    Gaussian.append(fitAndPredict(Gauss, traindata, trainlabels,testdata, testlabels))
    neigh = KNeighborsClassifier(n_neighbors=7)
    knn.append(fitAndPredict(neigh, traindata, trainlabels,testdata, testlabels))
    print "Test accuracies:"
    print "SVM:", sum(linearSVM)/float(len(linearSVM))
    print "GNB:", sum(Gaussian)/float(len(Gaussian))
    print "KNN:", sum(knn)/float(len(knn))

if __name__ == "__main__":
    data, labels = loadData("selectROI.mat")
    traindata,trainlabels,testdata,testlabels = shuffle(data, labels)
    #cross_validate(traindata,trainlabels)
    #clf, Gauss, neigh = cross_validate(traindata,trainlabels)
    trainAndtest(traindata, trainlabels, testdata,testlabels)

