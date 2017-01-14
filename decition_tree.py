from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO

allElectronicsData = open('/Users/utopia/Desktop/machine_learnning/data.csv','rb')
reader = csv.reader(allElectronicsData)
headers = reader.next()

print(headers) 

featureList= []
labelList = []

for row in reader:
    labelList.append(row[len(row) - 1])
    rowDict = {}
    for i in range(1, len(row)-1):
        rowDict[headers[i]] = row[i]

    featureList.append(rowDict)
print(featureList)

vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()
print("dummyX:" + str(dummyX))
print(vec.get_feature_names)

print("lableList:"+str(labelList))


lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print("dummY:" + str(dummyY))


clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(dummyX, dummyY)
print("clf:"+str(clf))


with open("allElectronicInformationGainOri.dot",'w') as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file = f)

oneRowX = dummyX[0, :]
print("oneRowX:"+str(oneRowX))

newRowx = oneRowX

newRowX[0] = 1
newRowX[2] = 0
print("newRows:"+str(newRowX))

predictedY = clf.predict(newRowX)
print("predictedY:"+str(predictedY))