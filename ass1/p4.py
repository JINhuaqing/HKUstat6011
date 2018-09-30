# coding:utf8
import numpy as np
import csv
from scipy import stats
import matplotlib.pyplot as plt

plt.switch_backend('agg')
plt.figure(figsize=(20, 10))

# load data
with open('./ass1/files/hw1q4.csv', 'r') as f:
    reader = csv.reader(f)
    data = [float(i[0]) for i in reader]

# get para
data = np.array(data)
n = len(data)
sigma0 = 10

# (a)
mupost = (1+sigma0**2*data.sum())/(1+n*sigma0**2)
sigmapost = np.sqrt(sigma0**2/(1+n*sigma0**2))
postdist = stats.norm(mupost, sigmapost)

# (b)
Pop0 = postdist.cdf(-0.2)
print(f'P(H0|y) is {Pop0}')
priordist = stats.norm(1, 10)
BF10 = (postdist.sf(-0.2)/postdist.cdf(-0.2))/(priordist.sf(-0.2)/priordist.cdf(-0.2))
print(f'BF10 is {BF10}')

# (c)

pvalue = 1 - stats.norm().cdf((data.mean()+0.2)/np.sqrt(1/n))
print(f'the p-value is {pvalue}')

# (e)
sigma0lst = np.array([0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10])
poplst = []
for sig in sigma0lst:
    mpost = (1+sig**2*data.sum())/(1+n*sig**2)
    sigpost = np.sqrt(sig**2/(1+n*sig**2))
    pdist = stats.norm(mpost, sigpost)
    Pop0 = pdist.cdf(-0.2)
    poplst.append(Pop0)

poparr = np.array(poplst)
logsigs = np.log(sigma0lst)
plt.title(r'P(H0|y) vs log($\sigma$0)')
plt.xlabel(r'logi($\sigma$0)')
plt.ylabel(r'P(H0|y)')
plt.plot(logsigs, poparr, '*-r')
plt.hlines(pvalue, -3, +3, linestyles='dashed', colors='c')
plt.savefig('./ass1/savedoc/q4fig.jpg')
