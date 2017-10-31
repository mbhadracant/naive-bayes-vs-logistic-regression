import numpy as np
from sklearn import linear_model, datasets

X = np.array([[6,180],[5.92,190],[5.58,170],[5.92,165],[5,100],[5.5,150],[5.42,130],[5.75,150]])
Y = np.array(['M','M','M','M','F','F','F','F'])
logreg = linear_model.LogisticRegression(max_iter=1000)
logreg.fit(X, Y)
print(logreg.predict([[5.2,130]]))

