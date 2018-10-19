# coding:utf8
import numpy as np
from scipy import stats as sts
import pickle
from pathlib import Path
from scipy.special import beta
from sympy import pi, symbols, sin, integrate


#plt.switch_backend('agg')
np.random.seed(0)
root = Path('./ass2/savedoc')


r = 6
def f(x):
    num = np.pi*(np.sin(np.pi*x))**r
    den = beta(0.5, (r+1)/2)
    return num/den
grv = sts.beta(a=2, b=4)
# compute the true value
x = symbols('x')
trueinte = float(integrate(x*sin(pi*x)**r, (x, 0, 1)))
trueinte = trueinte * np.pi / beta(0.5, (r+1)/2)
print(f'the True value of E_f(X) is {trueinte:.4g}')

# part(a) importance sampling
numa = 10000 
gsps = grv.rvs(numa)
ws = f(gsps)/grv.pdf(gsps)
ISinte = (gsps*ws).sum()/ws.sum()
print(f'the Ef(X) computed by importance sampling is {ISinte:.4g}')


# part (b) Independent Metropolis Algorithm
datalst = []
numb = 10000
urv = sts.uniform(0, 1)
xk = 0.5
for _ in range(numb):
    y = grv.rvs()
    u = urv.rvs()
    prob = np.min([f(y)*grv.pdf(xk)/f(xk)/grv.pdf(y), 1])
    if u <= prob:
        xk = y
    datalst.append(xk)
IMAinte = np.mean(datalst[:])
print(f'the Ef(X) computed by Independent Metropolis is {IMAinte:.4g}')


