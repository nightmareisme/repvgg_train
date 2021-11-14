import matplotlib.pyplot as plt
import numpy as np

loss =[]
file = open('acc.txt')
file = list(file)
file = file[0]
res = file.strip('[')
res = res.strip(']')
res = res.split(',')

for i in res:
    loss.append(float(i))

plt.title('trainloss')
plt.plot(np.arange(len(loss)),loss)
plt.legend(['Train Loss'], loc = 'upper right')
plt.show()
