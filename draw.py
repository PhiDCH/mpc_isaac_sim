from matplotlib import pyplot as plt
import numpy as np 
data = np.load('data.npy', allow_pickle=True)
plt.axes(projection='polar')
rads = np.linspace(0,2*np.pi,1972)
for i,rad in enumerate(rads):
    if data[i] < 2:
        plt.polar(rad, data[i],'g.')
plt.show()