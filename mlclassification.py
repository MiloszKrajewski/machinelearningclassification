import csv
import numpy
import scipy
import matplotlib.pyplot as plt
from scipy.constants.constants import alpha
from sklearn import preprocessing, neighbors

from sklearn import neighbors
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import knnplots

from sklearn.naive_bayes import GaussianNB

from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV

from sklearn.datasets import load_iris

import scipy.stats

fileName = "wdbc.csv"
file = open(fileName, "rU")
csvData = csv.reader(file)
dataList = list(csvData)
dataArray = numpy.array(dataList)

X = dataArray[:, 2:].astype(float)
y = dataArray[:, 1]
yFreq = scipy.stats.itemfreq(y)

le = preprocessing.LabelEncoder()
le.fit(y)
yT = le.transform(y)

XTrain, XTest, yTrain, yTest = train_test_split(X, yT)

# plt.bar(left=0, height=int(yFreq[0, 1]))
# plt.bar(left=1, height=int(yFreq[1, 1]))
# plt.show()

# correlationMatrix = numpy.corrcoef(X, rowvar=0)
# fig, ax = plt.subplots()
# heatmap = ax.pcolor(correlationMatrix, cmap=plt.cm.Greys)
# plt.show()

# plt.scatter(x=X[:, 0], y=X[:, 1], c=y)
# plt.show()

def mega_plot(X):
    _, cnt = X.shape

    plt.figure(figsize=(3*cnt, 3*cnt))
    for i in range(cnt):
        si = X[:, i]
        bins = numpy.linspace(min(si), max(si), 30)
        for j in range(cnt):
            plt.subplot(cnt, cnt, i + j*cnt + 1)
            if i == j:
                for c in ['M', 'B']:
                    plt.hist(si[y == c], alpha=0.4, color=c, bins=bins)
                    plt.xlabel(i)
            else:
                sj = X[:, j]
                plt.gca().scatter(si, sj, c=y, alpha=0.4)
                plt.xlabel(i)
                plt.ylabel(j)
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
    plt.tight_layout()
    plt.show()

# mega_plot(X[:, 0:6])

def KNN():
    nbrs = neighbors.NearestNeighbors(n_neighbors=3, algorithm="ball_tree").fit(X)
    distance, indices = nbrs.kneighbors(X)

    knn3 = neighbors.KNeighborsClassifier(n_neighbors=3).fit(X, yT)
    predict3 = knn3.predict(X)

    knn15 = neighbors.KNeighborsClassifier(n_neighbors=15).fit(X, yT)
    predict15 = knn15.predict(X)

    knnWD = neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance').fit(X, yT).predict(X)
    knnWU = neighbors.KNeighborsClassifier(n_neighbors=3, weights='uniform').fit(X, yT).predict(X)

    # print len(y[knnWD != knnWU])

    # print len(XTrain), len(XTest), len(yTrain), len(yTest)

    # Accuracy=(tp+tn)/total
    # Precision=tp/(tp+fp)
    # Recall=Sensitivity=tp/(tp+fn)
    # Specificity=tn/(fp+tn)

    predicted = neighbors.KNeighborsClassifier(n_neighbors=3).fit(XTrain, yTrain).predict(XTest)
    mat = metrics.confusion_matrix(yTest, predicted)

    print metrics.classification_report(yTest, predicted)
    print "accuracy: ", metrics.accuracy_score(yTest, predicted)

    # knnplots.plotaccuracy(XTrain, yTrain, XTest, yTest, 310)

    # knnplots.decisionplot(XTrain, yTrain, n_neighbors=3, weights='uniform')
    # knnplots.decisionplot(XTrain, yTrain, n_neighbors=3, weights='distance')

def GNB():
    model = GaussianNB().fit(XTrain, yTrain)
    predicted = model.predict(XTest)

    mat = metrics.confusion_matrix(yTest, predicted)

    print metrics.classification_report(yTest, predicted)
    print "accuracy: ", metrics.accuracy_score(yTest, predicted)

def CV():
    knn3 = neighbors.KNeighborsClassifier(n_neighbors=3)
    knn3scores = cross_validation.cross_val_score(knn3, XTrain, yTrain, cv=3)
    print knn3scores
    print knn3scores.mean()
    print knn3scores.std()

    knn15 = neighbors.KNeighborsClassifier(n_neighbors=15)
    knn15scores = cross_validation.cross_val_score(knn15, XTrain, yTrain, cv=5)
    print knn15scores
    print knn15scores.mean()
    print knn15scores.std()

    gnb = GaussianNB()
    gnbscores = cross_validation.cross_val_score(gnb, XTrain, yTrain, cv=5)
    print gnbscores
    print gnbscores.mean()
    print gnbscores.std()

def L28():
    knn3 = neighbors.KNeighborsClassifier(n_neighbors=3)
    knn15 = neighbors.KNeighborsClassifier(n_neighbors=15)
    gnb = GaussianNB()

    knn3means, knn3stddev = [], []
    knn15means, knn15stddev = [], []
    gnbmeans, gnbstddev = [], []

    ks = range(2, 21)

    for k in ks:
        knn3scores = cross_validation.cross_val_score(knn3, XTrain, yTrain, cv=k)
        knn15scores = cross_validation.cross_val_score(knn15, XTrain, yTrain, cv=k)
        gnbscores = cross_validation.cross_val_score(gnb, XTrain, yTrain, cv=k)
        knn3means.append(knn3scores.mean())
        knn3stddev.append(knn3scores.std())
        knn15means.append(knn15scores.mean())
        knn15stddev.append(knn15scores.std())
        gnbmeans.append(gnbscores.mean())
        gnbstddev.append(gnbscores.std())

    print knn3means
    print knn15means
    print gnbmeans

    predicted = model.predict(XTest)

    plt.plot(ks, knn3means, label='knn3', color='purple')
    plt.plot(ks, knn15means, label='knn15', color='yellow')
    plt.plot(ks, gnbmeans, label='gnb', color='blue')
    plt.legend(loc=3)
    plt.ylim(0.8, 1)
    plt.show()

    # plt.plot(ks, knn3stddev, label='knn3', color='purple')
    # plt.plot(ks, knn15stddev, label='knn15', color='yellow')
    # plt.plot(ks, gnbstddev, label='gnb', color='blue')
    # plt.legend(loc=3)
    # plt.ylim(0, 0.1)
    # plt.show()

def GS():
    parameters = [
        {
            'n_neighbors': [1, 3, 5, 10, 50, 100],
            'weights': ['uniform', 'distance'],
        }
    ]
    clf = GridSearchCV(neighbors.KNeighborsClassifier(), parameters, cv=10, scoring='f1')
    clf.fit(XTrain, yTrain)

    print clf.best_estimator_

def CV2():
    knn3 = neighbors.KNeighborsClassifier(n_neighbors=3)
    knn3scores = cross_validation.cross_val_score(knn3, XTrain, yTrain, cv=5)
    predicted = knn3.predict(XTest)

    mat = metrics.confusion_matrix(yTest, predicted)

    print knn3scores

    print metrics.classification_report(yTest, predicted)
    print "accuracy: ", metrics.accuracy_score(yTest, predicted)



# KNN()
# GNB()
#CV()
#L28()
#GS()
CV2()
