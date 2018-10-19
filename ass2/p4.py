# coding:utf8
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import pickle
from scipy import stats
from pathlib import Path

plt.switch_backend('agg')
root = Path('./ass2/savedoc')

# (a)
lmbd = 1
num = int(1e6)
datalsta = []

tk, rk = 1, 1
if not (root/'p4a.pkl').is_file():
    while len(datalsta)<num:
        lmbdk = lmbd*tk
        uvalue = stats.uniform(loc=np.exp(-lmbdk), scale=1-np.exp(-lmbdk)).rvs()
        rk = stats.poisson(mu=lmbdk).ppf(uvalue)
        tk = stats.gamma(a=rk, scale=1/lmbd).rvs()
        datalsta.append((rk, tk))
        print(f'we totally need {num} samples, now we get {len(datalsta)+1:>7}th')

    with open(root/'p4a.pkl', 'wb') as f:
        pickle.dump(datalsta, f)

with open(root/'p4a.pkl', 'rb') as f:
    data = pickle.load(f)

rks = [i[0] for i in data]
tks = [i[1] for i in data]
plt.figure(figsize=(20, 10)) 
plt.subplot(211)
plt.title('$R^*$')
plt.plot(rks, 'r', label='$R^*$')
plt.legend()
plt.subplot(212)
plt.title('T')
plt.plot(tks, 'g', label='T')
plt.legend()
plt.savefig(root/'p4a.jpg')
plt.close()

# (b)
datalstb = []
num = int(1e6)
xk, yk = 0.5, 0.5 
if not (root/'p4b.pkl').is_file():
    while len(datalstb)<num:
        yk = stats.norm(xk, 1).rvs()
        xk = stats.cauchy(loc=yk, scale=1).rvs()
        datalstb.append((xk, yk))
        print(f'we totally need {num} samples, now we get {len(datalstb):>7}th')

    with open(root/'p4b.pkl', 'wb') as f:
        pickle.dump(datalstb, f)

with open(root/'p4b.pkl', 'rb') as f:
    data = pickle.load(f)
xks = [i[0] for i in data]
yks = [i[1] for i in data]
plt.figure(figsize=(20, 10)) 
plt.subplot(211)
plt.title('X')
plt.plot(xks, 'g', label='X')
plt.legend()

plt.subplot(212)
plt.title('Y')
plt.plot(yks, 'r', label='Y')
plt.legend()
plt.savefig(root/'p4b.jpg')
plt.close()

