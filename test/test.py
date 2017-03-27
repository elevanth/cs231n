import numpy as np
x = [1,2,3,4,5,6,7,8,9,10]
y = np.array_split(x, 3)
print(y)
n = []
n = n.extend(y[0])
n = n.extend(y[1])
n = np.array(n)
print(n)
