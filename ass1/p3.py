#coding:utf8
import numpy as np
from scipy.special import gamma as gm
from scipy import stats
import argparse
from pathlib import Path
import csv
import pickle


def csvread(froot):
    with open(froot, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
    return rows

froot = Path('./ass1/files/hw1q3.csv')

# problem 3
rows = csvread(froot)
Yv = [float(i[0]) for i in rows]
arrY = np.array(Yv)

# part (c)
# setting
mu0 = 5
sigma0 = 10
xi0 = 0.01 
n = len(arrY)
size = 5000
datalst = []

# since we already know the conditionnal distribution, I use gibbs sampling here
# parameters
x0 = [mu0, 1]
def sigmac(tau):
    return np.sqrt(sigma0**2/(1+tau*n*(sigma0**2)))
def muc(tau):
    return (mu0+tau*sigma0**2*arrY.sum())/(1+tau*n*sigma0**2)
alphac = n/2 + xi0
def betac(mu):
    return xi0+0.5*np.sum((arrY-mu)**2)

# start simulation
muk, tauk = x0
flag = 0
while len(datalst)<size:
    rvmuc = stats.norm(muc(tauk), sigmac(tauk))
    muk = rvmuc.rvs()
    rvtauc = stats.gamma(alphac, scale=1/betac(muk))
    tauk = rvtauc.rvs()
    flag += 1
    if flag == 1000: 
        print(f'flag: {flag}, we get the first data')
        datalst.append([muk, tauk])
    if flag > 1000 and flag % 100 == 0:
        print(f'flag: {flag}, we get the  {(flag-1000)//100+1}th data')
        datalst.append([muk, tauk])
with open('./ass1/savedoc/p3c.pkl', 'wb') as f:
    pickle.dump(datalst, f)
    
