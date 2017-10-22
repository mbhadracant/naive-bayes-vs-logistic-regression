import matplotlib.pyplot as plt
X = range(1,11)
Y = []

for i in X:
	plt.plot([i],[i*i],'x')	
plt.show()
