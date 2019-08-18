# coding:utf8
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from scipy.special import roots_jacobi as jac
from scipy.special import roots_chebyt as che1 
from scipy.special import roots_chebyu as che2 
from scipy.special import roots_legendre as leg 

np.random.seed(0)
plt.switch_backend('agg')
root = Path('./ass3/savedoc/')


# Part (a)

def f(x):
    num = np.sin(x)**3 + 5
    den = np.sqrt(x**4 + 1)
    return num/den


xx = np.linspace(-1, 5, 10000)
plt.title('Integrand function')
plt.plot(xx, f(xx))
#if not (root/'p1.jpg').is_file():
#    plt.savefig(root/'p1.jpg')


# Part (b)
ns = [5, 10, 20]
b, a = 5, -1

def g(t):
    x = 0.5*(b-a)*t + 0.5*(a+b)
    return f(x)

def legw(x):
    return np.ones(len(x))

def che1w(x):
    return 1/np.sqrt(1-x**2)

def che2w(x):
    return np.sqrt(1-x**2)

def jacw(x, a=1, b=1):
    return (1-x)**a*(1+x)**b

def inte(nodes, ws, typ=0):
    funlst = [legw, che1w, che2w, jacw]
    ft = g(nodes)/funlst[typ](nodes)
    return (ws*ft).sum()*(b-a)/2

for n in ns:
    nodeout = 'nodes:   ' + '{:>8.5f} '*n
    wout = 'weights: ' + '{:>8.5f} '*n

    lnds, lws = leg(n)
    intt = inte(lnds, lws, 0)
    print('Legendre')
    print(f'n: {n}, integration: {intt:>8.5f}')
    print(nodeout.format(*lnds))
    print(wout.format(*lws))

    c1nds, c1ws = che1(n)
    intt = inte(c1nds, c1ws, 1)
    print('Chebyshev 1')
    print(f'n: {n}, integration: {intt:>8.5f}')
    print(nodeout.format(*c1nds))
    print(wout.format(*c1ws))

    c2nds, c2ws = che2(n)
    intt = inte(c2nds, c2ws, 2)
    print('Chebyshev 2')
    print(f'n: {n}, integration: {intt:>8.5f}')
    print(nodeout.format(*c2nds))
    print(wout.format(*c2ws))

    jnds, jws = jac(n, 1, 1)
    intt = inte(jnds, jws, 3)
    print('Jacobi')
    print(f'n: {n}, integration: {intt:>8.5f}')
    print(nodeout.format(*jnds))
    print(wout.format(*jws))

