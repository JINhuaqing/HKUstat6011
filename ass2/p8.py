# coding:utf8
import numpy as np
from scipy import stats as sts
import pickle
from pathlib import Path
import matplotlib.pyplot as plt


plt.switch_backend('agg')
np.random.seed(0)
root = Path('./ass2/savedoc')

with open('./ass2/files/beer_data.txt', 'r') as f:
    data = f.readlines()
data = [i.strip() for i in data]
data = [i.split('\t') for i in data]
dicdata = {}
for idx in range(1, 7):
    dicdata[idx] = [float(i[0]) for i in data[1:] if int(i[1])==idx]
dataarr = []
for value in dicdata.values():
    dataarr.append(value)
dataarr = np.array(dataarr)
n = len(data[1:])
k = 6
m = 8
# part (b)
a, b = 0.1, 0.1

def mumean(alpha):
    return (dataarr-alpha.reshape(-1, 1)).mean()
def muvar(sigma):
    return sigma**2/n

sigmashape = a+n/2
def sigmascale(mu, alpha):
    diff = dataarr - mu - alpha.reshape(-1, 1) 
    return b+(diff**2).sum()/2

taushape = a+k/2
def tauscale(alpha):
    return b+(alpha**2).sum()/2

def alphavari(sigma, tau):
    return (sigma**2*tau**2)/(sigma**2+tau**2*m)
def alphameani(sigma, tau, mu, idx):
    den = sigma**2 + m*tau**2
    ssum = (dataarr[idx-1]-mu).sum()
    return tau**2*ssum/den

numb = int(1e6)
burnnum = 5000

datalst = []
muk, sigmak, tauk = 0.1, 0.1, 0.1
alphak = np.zeros(k)+0.1
if not (root/'p8b.pkl').is_file():
    for idxx in range(1, numb+1):
        dt = []
        muk = sts.norm(mumean(alphak), np.sqrt(muvar(sigmak))).rvs()
        sigmak2 = sts.invgamma(a=sigmashape, scale=sigmascale(muk, alphak)).rvs()
        sigmak = np.sqrt(sigmak2)
        tauk2 = sts.invgamma(a=taushape, scale=tauscale(alphak)).rvs()
        tauk = np.sqrt(tauk2)
        for idx in range(1, k+1):
            alphak[idx-1] = sts.norm(alphameani(sigmak, tauk, muk, idx), alphavari(sigmak, tauk)).rvs()
        datalst.append([muk, sigmak2, tauk2, *list(alphak)]) 
        print(f'Part b, We totally need {numb} samples, and now we get {idxx}th')
    
    with open(root/'p8b.pkl', 'wb') as f:
        pickle.dump(datalst, f)
with open(root/'p8b.pkl', 'rb') as f:
    datalst = pickle.load(f)
datalstarr = np.array(datalst)


names = ['mu', 'sigma2', 'tau2', 'alpha1', 'alpha2', 'alpha3', 'alpha4', 'alpha5', 'alpha6']
fig, axes = plt.subplots(nrows=9, ncols=1, figsize=(10, 20))
fig.suptitle('trace plots part b')
for i in range(len(datalst[0])):
    data1 = datalstarr[:, i]
    axes[i].plot(data1, label=f'{names[i]}') 
    axes[i].set_ylabel(f'{names[i]}')
plt.savefig(root/'p8b.jpg')
plt.close()
output1 = '{:>10} '*len(names)
output2 = '{:>10.4g} '*len(names)
print(output1.format(*names))
print(output2.format(*list(datalstarr[burnnum:].mean(axis=0))))

# part (c)
a, b = 0, 0

def mumean(alpha):
    return (dataarr-alpha.reshape(-1, 1)).mean()
def muvar(sigma):
    return sigma**2/n

sigmashape = a+n/2
def sigmascale(mu, alpha):
    diff = dataarr - mu - alpha.reshape(-1, 1) 
    return b+(diff**2).sum()/2

taushape = a+k/2
def tauscale(alpha):
    return b+(alpha**2).sum()/2

def alphavari(sigma, tau):
    return (sigma**2*tau**2)/(sigma**2+tau**2*m)
def alphameani(sigma, tau, mu, idx):
    den = sigma**2 + m*tau**2
    ssum = (dataarr[idx-1]-mu).sum()
    return tau**2*ssum/den

numc = int(1e6)
burnnum = 5000

datalstc = []
muk, sigmak, tauk = 0.1, 0.1, 0.1
alphak = np.zeros(k)+0.1
if not (root/'p8c.pkl').is_file():
    for idxx in range(1, numc+1):
        dt = []
        muk = sts.norm(mumean(alphak), np.sqrt(muvar(sigmak))).rvs()
        sigmak2 = sts.invgamma(a=sigmashape, scale=sigmascale(muk, alphak)).rvs()
        sigmak = np.sqrt(sigmak2)
        tauk2 = sts.invgamma(a=taushape, scale=tauscale(alphak)).rvs()
        tauk = np.sqrt(tauk2)
        for idx in range(1, k+1):
            alphak[idx-1] = sts.norm(alphameani(sigmak, tauk, muk, idx), alphavari(sigmak, tauk)).rvs()
        datalstc.append([muk, sigmak2, tauk2, *list(alphak)]) 
        print(f'Part C, We totally need {numc} samples, and now we get {idxx}th')
    
    with open(root/'p8c.pkl', 'wb') as f:
        pickle.dump(datalstc, f)
with open(root/'p8c.pkl', 'rb') as f:
    datalstc = pickle.load(f)
datalstcarr = np.array(datalstc)


names = ['mu', 'sigma2', 'tau2', 'alpha1', 'alpha2', 'alpha3', 'alpha4', 'alpha5', 'alpha6']
fig, axes = plt.subplots(nrows=9, ncols=1, figsize=(10, 20))
fig.suptitle('trace plots part c')
for i in range(len(datalstc[0])):
    data1 = datalstcarr[:, i]
    axes[i].plot(data1, label=f'{names[i]}') 
    axes[i].set_ylabel(f'{names[i]}')
plt.savefig(root/'p8c.jpg')
print(output2.format(*list(datalstcarr[burnnum:].mean(axis=0))))
