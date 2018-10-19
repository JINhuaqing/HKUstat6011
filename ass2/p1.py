# coding:utf8
import numpy as np
from scipy import stats as sts
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns


np.random.seed(0)
plt.switch_backend('agg')
root = Path('./ass2/savedoc/')

# part (c) I don't know how to sample directly from X
a, b = 1, 1/2
numc = 10000
gcrv = sts.gamma(a=a, scale=1/b)
datalstc = []
for _ in range(numc):
    nu = gcrv.rvs()
    x = sts.norm(0, np.sqrt(nu)).rvs()
    datalstc.append(x)
datalstc = np.array(datalstc)
plt.figure(figsize=(20, 10))
plt.title('conditional sampling vs direct sampling')
plt.xlim([-10, 10])
sns.kdeplot(datalstc, color='red', label='sample from conditional distribution')
plt.savefig(root/'p1c.jpg')
plt.close()


# part (e) 
b = 2
nume = 10000
gerv = sts.invgamma(a=b/2, scale=b/2)
datalste = []
for _ in range(nume):
    nu = gerv.rvs()
    x = sts.norm(0, np.sqrt(nu)).rvs()
    datalste.append(x)

xrv = sts.t(2)
xdsps = xrv.rvs(nume)
datalste = np.array(datalste)

plt.figure(figsize=(20, 10))
plt.title('conditional sampling vs direct sampling')
plt.xlim([-10, 10])
sns.kdeplot(datalste, color='red', label='sample from conditional distribution')
sns.kdeplot(xdsps, color='blue', label='sample directly from distribution')
plt.savefig(root/'p1e.jpg')
plt.close()
