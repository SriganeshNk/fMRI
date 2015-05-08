__author__ = 'Sriganesh'

import scipy.io
import math
from sklearn import svm
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
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
    C_val = [0.0000001, 0.0005, 0.01, 2]
    C_label = ["linearSVM C = 0.0000001","linearSVM C = 0.0005","linearSVM C = 0.01","linearSVM C = 2" ]
    k_val = [1, 3, 7, 9, 15]
    k_label = ["1nn", "3nn", "7nn", "9nn", "15nn"]
    linearSVM, knn = [],[]
    while i < n:
        train, tlabel, test, label = CVsplit(traindata,trainlabels, i)
        L = []
        for k in C_val:
            clf = svm.LinearSVC(C=k)
            L.append(fitAndPredict(clf,train,tlabel,test,label))
        linearSVM.append(L)
        L = []
        for j in k_val:
            if j < len(train[0]):
                clf = KNeighborsClassifier(n_neighbors=j)
                L.append(fitAndPredict(clf,train,tlabel,test,label))
        if len(L) > 0:
            knn.append(L)
        i += 1
    C = C_val[getBestIndex(linearSVM, C_label)]
    k = k_val[getBestIndex(knn, k_label)]
    CVplot(linearSVM,C_label)
    CVplot(knn,k_label)
    return C, k


def getBestIndex(data, val):
    X = np.asarray(data)
    maximum, index = 0, 0
    for i in range(len(data[0])):
        temp = sum(X[:,i])/len(X[:,i])
        print "Average Accuracy Score for:", val[i], "is:", temp
        if temp > maximum:
            maximum = temp
            index = i
    return index


def CVplot(data, val):
    X = np.asarray(data)
    line, leg = [], []
    for i in range(len(data[0])):
        line1, = plt.plot(X[:,i], label = val[i])
        line.append(line1)
        leg.append(val[i])
    plt.legend(line,leg)
    plt.show()


def fitAndPredict(clf, traindata, trainlabels, testdata, testlabels):
    clf.fit(traindata, trainlabels)
    mylabel =[]
    for x in testdata:
        mylabel.append(clf.predict(x))
    #print "Accuracy:", accuracy_score(testlabels, mylabel)
    #print "(Precision, Recall, F1-Score) Label S:", precision_recall_fscore_support(testlabels, mylabel,labels=['S','P'], pos_label='P',average='binary')
    #print "(Precision, Recall, F1-Score) Label P:", precision_recall_fscore_support(testlabels, mylabel,labels=['S','P'], pos_label='S',average='binary')
    #print "----------------------"
    return accuracy_score(testlabels, mylabel)


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


def trainAndtest(traindata, trainlabels, testdata, testlabels, C_val, k_val):
    print "Test accuracies:"
    clf = svm.LinearSVC(C=C_val)
    print "SVM Scores:"
    print fitAndPredict(clf, traindata, trainlabels,testdata, testlabels)
    Gauss = GaussianNB()
    print "GNB: Scores"
    print fitAndPredict(Gauss, traindata, trainlabels,testdata, testlabels)
    print "Nearest Neighbour Scores"
    neigh = KNeighborsClassifier(n_neighbors=k_val)
    print fitAndPredict(neigh, traindata, trainlabels,testdata, testlabels)


def getClassConditionalData(data, labels):
    dataP, dataS = [], []
    for i in range(len(data)):
        if labels[i] == 'P':
            dataP.append(data[i])
        if labels[i] == 'S':
            dataS.append(data[i])
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


def learnStructure(dataP, dataS, Pp, Ps, TAN= True):
    tempMatrix = [[0 for i in range(len(dataP))] for j in range(len(dataP))]
    for i in range(len(dataP)):
        for j in range(i+1, len(dataP)):
            temp = 0.0
            if np.corrcoef(dataP[i], dataP[j])[0][1] != 1.0:
                temp += Pp * math.log(1-((np.corrcoef(dataP[i], dataP[j])[0][1])**2))
            if np.corrcoef(dataS[i], dataS[j])[0][1] != 1.0:
                temp += Ps * math.log(1-((np.corrcoef(dataS[i], dataS[j])[0][1])**2))
            temp *= (0.5)
            tempMatrix[i][j] = temp
            #tempMatrix[j][i] = temp
    MaxG = nx.DiGraph()
    if TAN:
        G = nx.from_scipy_sparse_matrix(minimum_spanning_tree(csr_matrix(tempMatrix)))
        adjList = G.adj
        i = 0
        notReturnable = {}
        MaxG = getDirectedTree(adjList, notReturnable, MaxG, i)
    else:
        G = nx.Graph(np.asmatrix(tempMatrix))
        adjList = sorted([(u,v,d['weight']) for (u,v,d) in G.edges(data=True)], key=lambda x:x[2])
        i = 2
        MaxG = getDirectedGraph(adjList, MaxG, i)
    return MaxG


def getDirectedGraph(adjList, MaxG, k):
    finished = {}
    for (u,v,d) in adjList:
        if v not in finished:
            finished[v] = 1
            MaxG.add_edge(u,v)
        else:
            if finished[v] < k:
                finished[v] += 1
                MaxG.add_edge(u,v)
        #if not nx.is_directed_acyclic_graph(MaxG):
        #    MaxG.remove_edge(u,v)
    return MaxG


def getDirectedTree(adjList, notReturnable, MaxG, i):
    x = adjList[i]
    notReturnable[i] = 1
    L = []
    for y in x.keys():
        if y not in notReturnable:
            MaxG.add_edge(i, y)
            L.append(y)
    for y in L:
        MaxG = getDirectedTree(adjList,notReturnable,MaxG, y)
    return MaxG


def getVariables(Tree, data, R = 0.0000001):
    Param = getParam(data)
    # Do topological sort to figure out nodes with least number of dependence
    Nodes = nx.topological_sort(Tree)
    Variables = {'TOPO': Nodes}
    for i in Nodes:
        mean, Var = Param[i]
        Mean = {'NodeMean': Param[i][0], 'Beta':[], 'ParentMean':[], 'Parent':[]}
        L = []
        for x in Tree.predecessors(i):
            if x != i:
                #Parentmean, ParentVar = Param[x]
                PCov = np.cov([data[i], data[x]])[0][1]
                PBeta = PCov/Param[x][1]
                Mean['Beta'].append(PBeta)
                Mean['ParentMean'].append(Param[x][0])
                Mean['Parent'].append(x)
                L.append(x)
        if len(L) > 0:
            Depend = [data[i]]
            num, dem = 0, 0
            for k in L:
                Depend = np.vstack((Depend, data[k]))
            Parent = Depend[1:]
            if len(Parent) > 2:
                num = np.linalg.det(np.cov(Depend)) + R
                dem = np.linalg.det(np.cov(Parent)) + R
            if len(Parent) == 2:
                num = np.linalg.det(np.cov(Depend)) + R
                dem = np.linalg.det(np.cov(Parent)) + R
            if len(Parent) == 1:
                num = np.linalg.det(np.cov(Depend)) + R
                dem = np.var(Parent) + R
            Var = num / dem
        Std = math.sqrt(Var)
        Variables[i] = (Mean, Std)
    return Variables


def infer(Variables, testdata):
    Prod = 1.0
    for i in Variables['TOPO']:
        mean = Variables[i][0]['NodeMean']
        for j in range(len(Variables[i][0]['Beta'])):
             mean += Variables[i][0]['Beta'][j] * (testdata[Variables[i][0]['Parent'][j]] - Variables[i][0]['ParentMean'][j])
        rv = norm(loc=mean, scale=Variables[i][1])
        pr = rv.pdf(testdata[i])
        if pr > 0.0001:
            Prod *= (pr/0.1)
    return Prod


def cv_TAN(traindata, trainlabels):
    n = 16
    i = 0
    TAN, KDTAN = [], []
    R = [0.01, 1, 10]
    R_label = ['regular=0.01', 'regular=1', 'regular=0.10']#, 'regular=4000']
    while i < n:
        L1, L2 = [], []
        train, tlabel, test, label = CVsplit(traindata, trainlabels, i)
        dataP, dataS = getClassConditionalData(train, tlabel)
        Pp, Ps = getClassProbability(tlabel)
        Tree1 = learnStructure(dataP, dataS, Pp, Ps, True)
        PVariable1 = [getVariables(Tree1,dataP, r) for r in R]
        SVariable1 = [getVariables(Tree1,dataS, r) for r in R]
        Tree2 = learnStructure(dataP, dataS, Pp, Ps, False)
        PVariable2 = [getVariables(Tree2,dataP, r) for r in R]
        SVariable2 = [getVariables(Tree2,dataS, r) for r in R]
        for j in range(len(R)):
            mylabel1, mylabel2 = [], []
            for x in test:
                PProd = Pp * infer(PVariable1[j], x)
                SProd = Ps * infer(SVariable1[j], x)
                temp = PProd + SProd
                PProd = PProd/temp
                SProd = SProd/temp
                if SProd >= PProd:
                    mylabel1.append('S')
                else:
                    mylabel1.append('P')
                PProd = Pp * infer(PVariable2[j], x)
                SProd = Ps * infer(SVariable2[j], x)
                temp = PProd + SProd
                PProd = PProd/temp
                SProd = SProd/temp
                if SProd >= PProd:
                    mylabel2.append('S')
                else:
                    mylabel2.append('P')
            L1.append(accuracy_score(label, mylabel1))
            L2.append(accuracy_score(label, mylabel2))
        TAN.append(L1)
        KDTAN.append(L2)
        i += 1
    T = R[getBestIndex(TAN, R_label)]
    K = R[getBestIndex(KDTAN, R_label)]
    CVplot(TAN,R_label)
    CVplot(KDTAN,R_label)
    return T,K

if __name__ == "__main__":
    data, labels = loadData("avgROI.mat")
    TANAcc, GaussAcc = [], []
    traindata, trainlabels, testdata, testlabels = shuffle(data, labels)
    C, k = cross_validate(traindata,trainlabels)
    T, K = cv_TAN(traindata,trainlabels)
    trainAndtest(traindata, trainlabels, testdata,testlabels, C, k)
    dataP, dataS = getClassConditionalData(traindata, trainlabels)
    Pp, Ps = getClassProbability(trainlabels)
    Tree = learnStructure(dataP, dataS, Pp, Ps, TAN=True)
    PVar = getVariables(Tree, dataP, R = T)
    SVar = getVariables(Tree, dataS, R = T)
    mylabel = []
    for x in testdata:
        PProd = Pp * infer(PVar, x)
        SProd = Ps * infer(SVar, x)
        temp = PProd + SProd
        PProd = PProd/temp
        SProd = SProd/temp
        if SProd >= PProd:
            mylabel.append('S')
        else:
            mylabel.append('P')
    print "TAN Accuracy:", accuracy_score(testlabels, mylabel)
    Tree = learnStructure(dataP, dataS, Pp, Ps, TAN=False)
    PVar = getVariables(Tree, dataP, R = K)
    SVar = getVariables(Tree, dataS, R = K)
    mylabel = []
    for x in testdata:
        PProd = Pp * infer(PVar, x)
        SProd = Ps * infer(SVar, x)
        temp = PProd + SProd
        PProd = PProd/temp
        SProd = SProd/temp
        if SProd >= PProd:
            mylabel.append('S')
        else:
            mylabel.append('P')
    print "KDTAN Accuracy:", accuracy_score(testlabels, mylabel)