# coding:utf8
import numpy as np
from scipy import stats
from scipy.special import gamma
import matplotlib.pyplot as plt
from pathlib import Path
import pickle


np.random.seed(0)
root = Path('./ass2/savedoc')
if not root.is_dir():
    root.mkdir()


C = np.sqrt(np.pi/2)/gamma(1.5)
t2 = stats.t(2)
laplace = stats.laplace()
plt.switch_backend('agg')

# part (b)
x = np.linspace(-10, 10, 1000)
plt.figure(figsize=(20, 10))
plt.title("t2 distribution with C vs laplace distribution")
plt.plot(x, laplace.pdf(x), 'r', label='laplace distribution')
plt.plot(x, C*t2.pdf(x), 'g', label='t2 distribution with C')
plt.legend()
plt.savefig(root/'p2b.jpg')
plt.close()


# part (c)
num = 10000
allnum = int(5e4)
u01 = stats.uniform(0,1)
therate = 1/C
f = laplace.pdf
g = t2.pdf
gs = t2.rvs(allnum)
us = u01.rvs(allnum)
probs = f(gs)/C/g(gs)
restsamples = gs[us<=probs]
print(f'theoretical rejection rate is {1-therate:>10.6f}')
print(f' experiment rejection rate is {1-len(restsamples)/allnum:>10.6f}')


# part (d)
gs1000 = gs[:1000]
us1000 = us[:1000]
probs1000 = probs[:1000]
x1, x2 = gs1000[us1000<=probs1000], gs1000[us1000>probs1000]
plt.figure(figsize=(20, 10))
plt.xlim([-10, 10])
plt.plot(x1, g(x1), 'o', label='accepted samples')
plt.plot(x2, g(x2), 'x', label='rejected samples')
plt.plot(x, laplace.pdf(x), 'r', label='laplace distribution')
plt.plot(x, t2.pdf(x), 'g', label='t2 distribution')
plt.legend()
plt.savefig(root/'p2d.jpg')

# part (e)
rg = 10
#norm = stats.
