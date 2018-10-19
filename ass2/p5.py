# coding:utf8
import numpy as np
from scipy import stats as sts
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import pickle


plt.switch_backend('agg')
root = Path('./ass2/savedoc/')
np.random.seed(0)
# part (a)
with open('./ass2/files/hw2q5.csv') as f:
    x = csv.reader(f)
    data = list(x)

ys = np.array([float(i[0]) for i in data[1:]]) 

n = len(ys)
numprefix = 1/3*(2*np.pi)**(-n/2)*np.sqrt(9/(1+9*n))
numinexp = - (9*(1+9*n)*(ys**2).sum()-(9*ys.sum())**2)/(18*(1+9*n))
numinte = numprefix*np.exp(numinexp)

# MC integration
def lkf(theta):
    prefix = (2*np.pi)**(-n/2)
    inexp = -((ys-theta.reshape(-1, 1))**2).sum(axis=1)/2
    return prefix*np.exp(inexp)

nrv = sts.norm(0, 3)
samples = nrv.rvs(1000000)
mcinte = lkf(samples).mean()
print(f'the numerical integration is {numinte:.4e}')
print(f'the Monte Carlo integration is {mcinte:.4e}')

# part (b)
pmu = 9*ys.sum()/(9*n+1)
psigma = np.sqrt(9/(9*n+1))
pnrv = sts.norm(pmu, psigma)
numB = int(1e3)
Bs = np.arange(1, numB+1)
def phaty(B):
    thetas = pnrv.rvs(B)
    pyps = lkf(thetas)
    inpyps = 1/pyps
    return 1/inpyps.mean()

if not (root/'p5b.pkl').is_file():
    phatys = [phaty(B) for B in Bs]
    with open(root/'p5b.pkl', 'wb') as f:
        pickle.dump(phatys, f)

with open(root/'p5b.pkl', 'rb') as f:
    phatys = pickle.load(f)
plt.figure(figsize=(20, 10))
plt.plot(Bs, phatys, 'g')
plt.xlabel('B')
plt.ylabel('$\hat{p}(y)$')
if not (root/'p5b.jpg').is_file():
    plt.savefig(root/'p5b.jpg')
plt.close()
print(f'the harmonic estimator is {np.mean(phatys):.4g}')
