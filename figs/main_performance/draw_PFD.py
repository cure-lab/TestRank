import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from scipy.interpolate import make_interp_spline as spline

# the value in result is pfd value. we normalzie it to tpf value by deviding the pdf value to ideal pdf value
result = pd.read_csv('result.csv')
print (result)

def normalize_gap(inputs):
    ideal = result['Ideal']
    return 100*inputs/ideal

x = result['Budget']
x_smooth = np.linspace(x.min(),x.max(),300)

def y_smooth(y):
    global x
    return spline(x,y)(x_smooth)

colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02' ]
# model A
plt.figure()
plt.plot(x_smooth, y_smooth(normalize_gap(result['Ours'])), label='Ours', color=colors[0]) 
plt.scatter(x, normalize_gap(result['Ours']), marker='o', color=colors[0])
plt.plot(x_smooth, y_smooth(normalize_gap(result['DeepGini'])), label='DeepGini', color=colors[1]) 
plt.scatter(x, normalize_gap(result['DeepGini']), marker='o', color=colors[1])
plt.plot(x_smooth, y_smooth(normalize_gap(result['MCP'])),label='MCP', color=colors[2]) 
plt.scatter(x, normalize_gap(result['MCP']), marker='o', color=colors[2])
plt.plot(x_smooth, y_smooth(normalize_gap(result['DSA'])), label='DSA', color=colors[3]) 
plt.scatter(x, normalize_gap(result['DSA']), marker='o', color=colors[3])
plt.plot(x_smooth, y_smooth(normalize_gap(result['Uncertainty'])), label='Uncertainty', color=colors[4]) 
plt.scatter(x, normalize_gap(result['Uncertainty']), marker='o', color=colors[4])
plt.plot(x_smooth, y_smooth(normalize_gap(result['Budget'])), label='Random', color=colors[5]) # to represent random selection
plt.scatter(x, normalize_gap(result['Budget']), marker='o', color=colors[5])
plt.legend(fontsize=13)
plt.tick_params(axis='both', labelsize=14)#刻度大小
plt.grid(linestyle='--')
plt.savefig('PDF_diff.pdf')