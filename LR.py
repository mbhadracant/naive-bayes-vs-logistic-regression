import numpy as np
import random

X = np.array([[6,180,12],[5.92,190,11],[5.58,170,12],[5.92,165,10],[5,100,6],[5.5,150,8],[5.42,130,7],[5.75,150,9]])
Y = np.array([0,0,0,0,1,1,1,1])

def predict(row, pair):
    sum = 0

    for i in range(0,len(row)):
        sum += ((row[i] * pair[0][i]) - pair[1])
    return 1.0 if sum > 0 else 0.0


def test():
    t = random.randint(1, 50)

    w = np.zeros(len(X[0]))
    a = 0.1
    while True:
        temp = np.copy(w)
        for i in range(0,len(X)):
            example = X[i]
            label = Y[i]
            prediction = predict(example, (w,t))
            for j in range(0,len(example)):
                w[j] = w[j] - a * (prediction - label) * example[j]
            t = t + a * (prediction - label)

        if np.array_equal(temp, w):
            break

    return (w,t)

pair = test()
result = 'M' if predict([6,180,10], pair) == 0 else 'F'
print(result)