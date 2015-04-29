__author__ = 'Sriganesh'

import scipy.io
import math
from sklearn import svm
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
import operator
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx
import random
from scipy.stats import norm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

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
    i = 0
    k_val = [1, 3, 7, 9, 15]
    linearSVM, Gauss, knn = [],[],[]
    while i < n:
        train, tlabel, test, label = CVsplit(traindata,trainlabels, i)
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
        line, = plt.plot(X[:,i], label = str(k_val[i])+"nn")
        line3.append(line)
        leg3.append(str(k_val[i])+"nn")
    plt.legend(line3,leg3)
    plt.show()

def fitAndPredict(clf, traindata, trainlabels, testdata, testlabels):
    clf.fit(traindata, trainlabels)
    i , correct = 0, 0
    for x in testdata:
        if testlabels[i] == clf.predict(x):
            correct += 1
        i += 1
    return float(correct)/float(len(testlabels))

def shuffle(data, labels):
    start = len(data) - (6*len(data)/10)
    end = len(data) - (5*len(data)/10)
    print start, end
    x = [i for i in range(len(data))]
    random.shuffle(x)
    traindata = [data[i] for i in x[:start]]
    traindata.extend([data[i] for i in x[end:]])
    trainlabels = [labels[i] for i in x[:start]]
    trainlabels.extend([labels[i] for i in x[end:]])
    testdata = [data[i] for i in x[start:end]]
    testlabels = [labels[i] for i in x[start:end]]
    return (traindata,trainlabels,testdata,testlabels)

def trainAndtest(traindata, trainlabels, testdata, testlabels):
    clf = svm.SVC(C=5000,kernel="linear")
    linearSVM = fitAndPredict(clf, traindata, trainlabels,testdata, testlabels)
    Gauss = GaussianNB()
    Gaussian = fitAndPredict(Gauss, traindata, trainlabels,testdata, testlabels)
    neigh = KNeighborsClassifier(n_neighbors=7)
    knn = fitAndPredict(neigh, traindata, trainlabels,testdata, testlabels)
    print "Test accuracies:"
    print "SVM:", linearSVM
    print "GNB:", Gaussian
    print "KNN:", knn


def getClassConditionalData(data):
    dataP, dataS = [], []
    for i in range(len(data)):
        if labels[i] == 'P':
            dataP.append([data[i][j] for j in range(len(data[i]))])
        if labels[i] == 'S':
            dataS.append([data[i][j] for j in range(len(data[i]))])
    dataP = np.transpose(dataP)
    dataS = np.transpose(dataS)
    return dataP, dataS

def getClassProbability(labels):
    s, p = 0, 0
    for x in labels:
        if x == 'S':
            s += 1
        if x == 'P':
            p += 1
    Pp = float(p)/float(len(labels))
    Ps = float(s)/float(len(labels))
    return Pp, Ps

def getParam(data):
    Params = []
    for x in data:
        Mu = np.mean(x)
        Sigma = np.var(x)
        Params.append((Mu, Sigma))
    return Params

def CVsplit(traindata, trainlabels, i=5):
    n = 16
    limit = len(traindata)/n
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
    test = traindata[start:end]
    label = trainlabels[start:end]
    return train,tlabel,test,label


def maximum_spanning_tree(G, TAN = False):
    Edges = [(u,v,d['weight']) for (u,v,d) in G.edges(data=True)]
    Edges.sort(key=lambda x:x[2])
    MaxG = nx.DiGraph()
    for x in Edges:
        MaxG.add_edges(x[0],x[1])
        if not TAN and not nx.is_directed_acyclic_graph(G):
            MaxG.remove_edge(x[0],x[1])
        if TAN and not nx.is_tree(G):
            MaxG.remove_edge(x[0],x[1])
    print [(u,v) for (u,v,w) in Edges]
    print [(u,v) for (u,v) in MaxG.edges()]
    return G


def learnStructure(dataP, dataS, Pp, Ps, R = 0.005):
    tempMatrix = [[0 for i in range(len(dataP))] for j in range(len(dataP))]
    for i in range(len(dataP)):
        for j in range(i+1, len(dataP)):
            try:
                if i != j:
                    temp = Pp * math.log(1-((np.corrcoef(dataP[i], dataP[j])[0][1])**2))
                    temp += Ps * math.log(1-((np.corrcoef(dataS[i], dataS[j])[0][1])**2))
                    temp *= (-0.5)
                    tempMatrix[i][j] = temp
                    #tempMatrix[j][i] = temp
            except ValueError:
                print "DATA1:", dataP[i]
                print "DATA2:", dataP[j]
                print "Correlation coefficient:", np.corrcoef(dataP[i], dataP[j])[0][1]
    #print tempMatrix
    G = nx.DiGraph(np.asmatrix(tempMatrix))
    G = maximum_spanning_tree(G, True)
    #print G.adjacency_list()
    #G = nx.DiGraph(minimum_spanning_tree(csr_matrix(tempMatrix)).toarray().astype(float))
    return G

def infer(Tree, data, testdata):
    Param = getParam(data)
    Prod = 1
    # Do topological sort to figure out nodes with least number of dependence
    Nodes = nx.topological_sort(Tree)
    print Nodes
    for i in range(len(Nodes)):
        mean, Var = Param[i]
        Sum = 0
        L = []
        for x in Tree.predecessors(Nodes[i]):
            if x != i:
                Parentmean, ParentVar = Param[x]
                PCov = np.cov([data[i], data[x]])[0][1]
                PBeta = PCov/ParentVar
                Sum += PBeta * (testdata[x] - Parentmean)
                L.append(x)
        mean += Sum
        if len(L) > 0:
            Parent = [data[i]]
            for k in L:
                Parent.append(data[k])
            if len(Parent) > 2:
                Var = np.linalg.det(np.cov(Parent)) / np.linalg.det(np.cov(Parent[1:]))
            else:
                Var = np.linalg.det(np.cov(Parent)) / Param[L[0]][1]
        Prod *= norm.pdf(testdata[i], loc=mean, scale=Var)
    return Prod

def cv_TAN(traindata, trainlabels, R=0.005):
    n = 16
    i = 0
    Accuracy = []
    while i < n:
        train, tlabel, test, label = CVsplit(traindata, trainlabels, i)
        dataP, dataS = getClassConditionalData(train)
        Pp, Ps = getClassProbability(tlabel)
        Tree = learnStructure(dataP, dataS, Pp, Ps, R)
        mylabel = []
        for x in test:
            PProd = Pp * infer(Tree, dataP, x)
            SProd = Ps * infer(Tree, dataS, x)
            if SProd >= PProd:
                mylabel.append('S')
            else:
                mylabel.append('P')
        Accuracy.append(accuracy_score(label, mylabel))
        #print "Accuracy:", float(correct)/float(len(label))
        #print "Accuracy:", accuracy_score(label, mylabel)
        #print "(Precision, Recall, F1-Score)", precision_recall_fscore_support(label, mylabel)
        i += 1
    return sum(Accuracy)/len(Accuracy)


if __name__ == "__main__":
    data, labels = loadData("avgROI.mat")
    TANAcc, GaussAcc = [], []
    for y in range(1):
        traindata, trainlabels, testdata, testlabels = shuffle(data, labels)#CVsplit(data, labels)
        #temp = cv_TAN(traindata,trainlabels)
        #print "CV", temp
        dataP, dataS = getClassConditionalData(traindata)
        Pp, Ps = getClassProbability(trainlabels)
        Tree = learnStructure(dataP, dataS, Pp, Ps)
        mylabel = []
        for x in testdata:
            PProd = Pp * infer(Tree, dataP, x)
            SProd = Ps * infer(Tree, dataS, x)
            if SProd >= PProd:
                mylabel.append('S')
            else:
                mylabel.append('P')
        TANAcc.append(accuracy_score(testlabels,mylabel))
        #print "Accuracy:", accuracy_score(testlabels, mylabel)
        #print "(Precision, Recall, F1-Score)", precision_recall_fscore_support(testlabels,mylabel)
        Gauss = GaussianNB()
        #print "Gaussian:", fitAndPredict(Gauss, traindata, trainlabels,testdata, testlabels)
        GaussAcc.append(fitAndPredict(Gauss, traindata, trainlabels, testdata, testlabels))
    print "TAN:", sum(TANAcc)/len(TANAcc)
    print "GAUSS:", sum(GaussAcc)/len(GaussAcc)
    #cross_validate(traindata,trainlabels)
    #clf, Gauss, neigh = cross_validate(traindata,trainlabels)
    #trainAndtest(traindata, trainlabels, testdata,testlabels)"""

