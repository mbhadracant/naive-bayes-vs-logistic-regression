import numpy as np
import matplotlib.pyplot as plt

arr1 = [1, 2, 3, 4, 5]
arr2 = [2, 3, 3, 4, 4]
labl = [0, 1, 1, 0, 0]
color= ['red' if l == 0 else 'green' for l in labl]
print(color)
plt.scatter(arr1, arr2, color=color)
plt.show()
