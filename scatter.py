import numpy as np
import matplotlib.pyplot as plt

arr1 = [6, 5.92, 5.58, 5.92, 5, 5.5, 5.42, 5.75]
arr2 = [180, 190, 170, 165, 100, 150, 130, 150]
labl = [1, 1, 1, 1, 0, 0, 0, 0]
color= ['red' if l == 0 else 'green' for l in labl]
plt.scatter(arr1, arr2, color=color)
plt.show()
