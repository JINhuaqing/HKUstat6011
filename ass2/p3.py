# coding:utf8
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path

plt.switch_backend('agg')

root = Path('./ass2/savedoc/')
if not root.is_dir():
    root.mkdir()

# part (a)
lognormrv = stats.lognorm(s=2, scale=1)
gammarv = stats.gamma(a=2, scale=2)
J = int(1e5)
m = int(J/10)
gsamples = lognormrv.rvs(J)
ws = gammarv.pdf(gsamples)/lognormrv.pdf(gsamples)
wsn = ws/ws.sum()
fsamples = np.random.choice(gsamples, m, False, p=wsn)

plt.figure(figsize=(20, 10))
xx = np.linspace(0, 10, 100000)
plt.title('gamma vs log normal vs SIR histogram')
plt.plot(xx, gammarv.pdf(xx), 'r', label='gamma distribution')
plt.plot(xx, lognormrv.pdf(xx), 'g', label='log normal distribution')
plt.hist(fsamples, bins=50, density=True)
plt.legend()
if not (root/'p3a.jpg').is_file():
    plt.savefig(root/'p3a.jpg')

# part (b)
xmode = 2
a, b = 1, 3
sncdf = stats.norm().cdf
tvalue = gammarv.cdf(b) - gammarv.cdf(a)
def h(x):
    return np.log(1/4*x*np.exp(-x/2))

def h2(x):
    return -x**(-2) 
def tmp(x):
    return (x-xmode)/(-h2(xmode))**(-0.5)

avalue = np.exp(h(xmode))*(2*np.pi/-h2(xmode))**(0.5)*(sncdf(tmp(b))-sncdf(tmp(a)))
print(f'the true value is {tvalue:.5f}')
print(f'the approx value is {avalue:.5f}')
