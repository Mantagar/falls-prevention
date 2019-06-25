from model_utils import *
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

dataPaths, _, _ = loadDataPaths("data/training_set.txt")

testDataPaths, _, _ = loadDataPaths("data/test_set.txt")
testDataPaths2, _, _ = loadDataPaths("data/validation_set.txt")
testDataPaths += testDataPaths2

#NONSEQUENTIAL
print("Preparing traing data")
batcher = Batcher(dataPaths, 1, 1)
X = []
Y = []
while batcher.hasNextBatch():
  x, y = batcher.nextBatch()
  x = list(x.reshape(2).numpy())
  y = list(y.numpy()[0])
  X.append(x)
  Y.append(y)
  
print("Preparing test data")
batcher = Batcher(testDataPaths, 1, 1)
tX = []
tY = []
while batcher.hasNextBatch():
  x, y = batcher.nextBatch()
  x = list(x.reshape(2).numpy())
  y = list(y.numpy()[0])
  tX.append(x)
  tY.append(y)
'''
#SUMMED SEQUENTIAL
print("Preparing traing data")
batcher = Batcher(dataPaths, 300, 1)
X = []
Y = []
while batcher.hasNextBatch():
  x, y = batcher.nextBatch()
  x = list(x.reshape(300,2).numpy())
  x = [v[0]+v[1] for v in x]
  y = list(y.numpy()[0])
  X.append(x)
  Y.append(y)
  
print("Preparing test data")
batcher = Batcher(testDataPaths, 300, 1)
tX = []
tY = []
while batcher.hasNextBatch():
  x, y = batcher.nextBatch()
  x = list(x.reshape(300,2).numpy())
  x = [v[0]+v[1] for v in x]
  y = list(y.numpy()[0])
  tX.append(x)
  tY.append(y)
'''
  
  
print("Creating classifiers")
dtc = tree.DecisionTreeClassifier()
dtc = dtc.fit(X,Y)

knn = KNeighborsClassifier(n_neighbors=2001)
knn = knn.fit(X,Y)

gnb = GaussianNB()
gnb = gnb.fit(X,Y)

rfc = RandomForestClassifier(n_estimators=101)
rfc = rfc.fit(X,Y)

svm = LinearSVC(random_state=0, tol=1e-5)
svm = svm.fit(X, Y)

print("Evaluating models")
print("DTC:",dtc.score(tX,tY))
print("KNN:",knn.score(tX,tY))
print("GNB:",gnb.score(tX,tY))
print("RFC:",rfc.score(tX,tY))
print("SVM:",svm.score(tX,tY))
  