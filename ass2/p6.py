# coding:utf8
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from scipy import stats as sts

np.random.seed(0)
plt.switch_backend('agg')
root = Path('./ass2/savedoc/')

sigmas = [0.1, 0.5, 1, 2, 5]
num = int(1e4)
tpdf = sts.t(1).pdf
urv = sts.uniform(0, 1)

datadic = {}
accrates = {}
if not (root/'p6.pkl').is_file():
    for sigma in sigmas:
        nrv = sts.norm(0, sigma)
        xk = 1
        datalst = []
        us = []
        probs = []
        while len(datalst)<num:
            eps = nrv.rvs()
            u = urv.rvs()
            y = xk + eps
            prob = np.min([tpdf(y)/tpdf(xk), 1])
            if u <= prob:
                xk = y
            datalst.append(xk)
            us.append(u)
            probs.append(prob)
            print(f'sigma is {sigma:>5}, we need {num} samples, and now we get {len(datalst):>5}th')

        us = np.array(us)
        probs = np.array(probs)
        accps = us <= probs
        accrate = accps.sum()/len(accps)
        datadic[sigma] = datalst
        accrates[sigma] = accrate
        print(f'sigma is {sigma}, empirical acceptance rate is {accps.sum()/len(accps):.3f}')
    with open(root/'p6.pkl', 'wb') as f:
        pickle.dump([datadic, accrates], f)

with open(root/'p6.pkl', 'rb') as f:
    datadic, accrates = pickle.load(f)
output1 = 'sigma:      '+' {:.4f}'*len(sigmas)
output2 = 'accept rate:'+' {:.4f}'*len(sigmas)
print(output1.format(*sigmas))
print(output2.format(*list(accrates.values())))

fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(30, 10))

fig.suptitle('trace plots of different sigmas')
for i, data in enumerate(datadic.items()):
    sigma, datalst = data
    axes[i].plot(datalst, label=f'sigma {sigma}')
    axes[i].set_ylabel(f'sigma {sigma}')

for sigma,  datalst in datadic.items():
    axes[-1].plot(datalst, label=f'sigma {sigma}')
axes[-1].set_ylabel(f'all sigmas ')
plt.legend()
plt.savefig(root/'p6.jpg')
