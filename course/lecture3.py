# coding:utf8
import numpy as np
from scipy import stats
from scipy.special import beta
import matplotlib.pyplot as plt
from sympy import *
from pathlib import Path

plt.switch_backend('agg')
np.random.seed(0)
num = int(1e6)
root = Path('./course/savedoc')
if not root.is_dir():
    root.mkdir()

## Monte Carlo Integration
print('Monte Carlo Method')
# (1)
def f(x):
    out =  np.cos(50*x)+np.sin(20*x)
    return out**2

urvs = stats.uniform(0, 1)
simudata1 = urvs.rvs(num)
simuout = f(simudata1).mean()

# true value
x = symbols('x')
truevalue = integrate((cos(50*x)+sin(20*x))**2, (x, 0, 1))
truevalue = float(truevalue)

print(f'the simulation value is {simuout:.4f}')
print(f'the true value is {truevalue:.4f}')


# (2)
a, b = 4, 19
y = symbols('y')
ty = integrate(1/y, (y, a, b))
ty = float(ty)
yurvs = stats.uniform(a, b-a)
ysimudata = yurvs.rvs(num)
ysimuout = (b-a)*((1/ysimudata).mean())

print(f'the simulation value is {ysimuout:.4f}')
print(f'the true value is {ty:.4f}')

## Inversion Method
# exponential distribution
print('Inversion Method')
lbd = 3
def revfn(x, lbd):
    return -np.log(1-x)/lbd
ervs = stats.expon(scale=1/lbd)
tdis = ervs.rvs(num)

urvs = stats.uniform(0, 1)
sdis = urvs.rvs(num)
sdis = revfn(sdis, lbd)
plt.hist(tdis, color='red', label='true distribution')
plt.hist(sdis, color='green', label='simulation distribution')
plt.savefig(root/'im.jpg')
print(f'mean of true vaule is {tdis.mean():.5f}, std of true value is {tdis.mean():.5f}, lambda is {lbd}')
print(f'mean of simulation vaule is {sdis.mean():.5f}, std of simulation value is {sdis.std():.5f}')


## sampling/Importance Resampling
print('SIR method')
r = 6
J = int(2e3)
m = int(J/10)
def f(x):
    numer = np.pi*(np.sin(np.pi*x))**r
    den = beta(1/2, (r+1)/2)
    return numer/float(den)
grv = stats.beta(a=2, b=4)
gsamples = grv.rvs(J)
ws = f(gsamples)/grv.pdf(gsamples)
wsn = ws/ws.sum()
fsamples = np.random.choice(gsamples, m, replace=False, p=wsn)

plt.figure(figsize=(20, 10))
px = np.linspace(0, 1, 10000)
plt.plot(px, f(px), 'r')
plt.hist(fsamples, bins=80, density=True)
if not (root/'sir.jpg').is_file():
    plt.savefig(root/'sir.jpg')
plt.close()
