import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

X = np.array([[6,180,12],[5.92,190,11],[5.58,170,12],[5.92,165,10],[5,100,6],[5.5,150,8],[5.42,130,7],[5.75,150,9]])
Y = np.array(['M','M','M','M','F','F','F','F'])
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X, Y)
print(logreg.predict([[6,130,8]]))
