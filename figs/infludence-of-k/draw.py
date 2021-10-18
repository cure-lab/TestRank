import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from scipy.interpolate import make_interp_spline as spline


result = pd.read_csv('result.csv')
result.set_index('Dataset', inplace=True)
print (result)


x = result.loc['k', :]
print (x.min(), x.max())
x_smooth = np.linspace(x.min(),x.max(), 300)
print ('x smooth: ', x_smooth.size)

print (result.loc['A', :])

def y_smooth(y):
    global x, x_smooth
    print (x.size, y.shape)
    return spline(x,y)(x_smooth)


colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02' ]
plt.figure()
plt.plot(x_smooth, y_smooth(result.loc['A', :]), label='Model A', color=colors[0]) 
plt.scatter(x, result.loc['A', :], marker='o', color=colors[0])
plt.plot(x_smooth, y_smooth(result.loc['B', :]), label='Model B', color=colors[1]) 
plt.scatter(x, result.loc['B', :], marker='o', color=colors[1])
plt.plot(x_smooth, y_smooth(result.loc['C', :]),label='Model C', color=colors[2]) 
plt.scatter(x, result.loc['C', :], marker='o', color=colors[2])
plt.ylim(0, 100)
plt.xlabel('Number of Neighbors', fontsize=15)
plt.ylabel('ATPF values (%)', fontsize=15)
plt.legend(fontsize=14)
plt.tick_params(axis='both', labelsize=15)
plt.grid(linestyle='--')
plt.savefig('influence-k-neighbors.pdf')