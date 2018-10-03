# coding:utf8
import numpy as np
import csv
from scipy import stats
import pickle
import prettytable as pt


# load data
with open('./ass1/files/hw1q5.csv') as f:
    f = csv.reader(f)
    data = list(f)

X1s = np.array([float(i[0]) for i in data[1:]])
X2s = np.array([float(i[1]) for i in data[1:]])
X3s = X1s * X2s
Ys = np.array([float(i[2]) for i in data[1:]])
n = len(Ys)
xi0 = 0.1
size = 10000
step = 50 
initnum = 5000

# (d) 
# gibbs sampling
initvalue = [0, 0, 0, 0, 1]
def mubeta1(beta0, beta2, beta3, sigma):
    return np.sum((Ys-X2s*beta2-X3s*beta3-beta0)*X1s)/(sigma**2/100+np.sum(X1s**2))
def mubeta2(beta0, beta1, beta3, sigma):
    return np.sum((Ys-X1s*beta1-X3s*beta3-beta0)*X2s)/(sigma**2/100+np.sum(X2s**2))
def mubeta3(beta0, beta1, beta2, sigma):
    return np.sum((Ys-X1s*beta1-X2s*beta2-beta0)*X3s)/(sigma**2/100+np.sum(X3s**2))
def mubeta0(beta1, beta2, beta3, sigma):
    return np.sum((Ys-X1s*beta1-X2s*beta2-X3s*beta3))/(sigma**2/100+n)
def sigmabeta1(sigma):
    return np.sqrt(sigma**2/(sigma**2/100+np.sum(X1s**2)))
def sigmabeta2(sigma):
    return np.sqrt(sigma**2/(sigma**2/100+np.sum(X2s**2)))
def sigmabeta3(sigma):
    return np.sqrt(sigma**2/(sigma**2/100+np.sum(X3s**2)))
def sigmabeta0(sigma):
    return np.sqrt(sigma**2/(sigma**2/100+n))
def betas2(beta0, beta1, beta2, beta3):
    return xi0 + np.sum((Ys-beta0-beta1*X1s-beta2*X2s-beta3*X3s)**2)/2
alphas2 = xi0 + n/2

# start
datalst = []
flag = 0
beta0k, beta1k, beta2k, beta3k, tauk = initvalue
while len(datalst)<size:
    beta0rv = stats.norm(mubeta0(beta1k, beta2k, beta3k, np.sqrt(tauk)), sigmabeta0(np.sqrt(tauk)))
    beta0k = beta0rv.rvs()
    beta1rv = stats.norm(mubeta1(beta0k, beta2k, beta3k, np.sqrt(tauk)), sigmabeta1(np.sqrt(tauk)))
    beta1k = beta1rv.rvs()
    beta2rv = stats.norm(mubeta2(beta0k, beta1k, beta3k, np.sqrt(tauk)), sigmabeta2(np.sqrt(tauk)))
    beta2k = beta2rv.rvs()
    beta3rv = stats.norm(mubeta3(beta0k, beta1k, beta2k, np.sqrt(tauk)), sigmabeta3(np.sqrt(tauk)))
    beta3k = beta3rv.rvs()
    s2rv = stats.invgamma(alphas2, scale=betas2(beta0k, beta1k, beta2k, beta3k))
    tauk = s2rv.rvs()
    flag += 1
    if flag == initnum: 
        print(f'flag: {flag:>10}, we get the first data')
        datalst.append((beta1k, beta1k, beta2k, beta3k, tauk))
    if flag > initnum and flag % step == 0:
        print(f'flag: {flag:>10}, we get the  {(flag-initnum)//step+1:>7}th data')
        datalst.append((beta0k, beta1k, beta2k, beta3k, tauk))

with open('./ass1/savedoc/p5d.pkl', 'wb') as f:
    pickle.dump(datalst, f)

arrdata = np.array(datalst)
means = arrdata.mean(axis=0)
variances = arrdata.var(axis=0)

# output
tb = pt.PrettyTable()
tb.field_names = ['type', 'beta0', 'beta1', 'beta2', 'beta3', 'sigma2']
tb.add_row(['means']+list(np.round(means, 4)))
tb.add_row(['variances']+list(np.round(variances, 4)))
print(tb)
