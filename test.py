import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import datasets, linear_model
from sklearn.naive_bayes import GaussianNB

iris = datasets.load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.5, random_state=0)
clf = linear_model.LogisticRegression(C=1e5,multi_class='multinomial')
clf.fit(x_train,y_train)
scores = cross_val_score(clf, iris.data, iris.target, cv=5)
print(clf.predict([[5.9,3.0,5.1,1.8]]))
print(clf.score(x_test,y_test))

