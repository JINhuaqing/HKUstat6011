#coding:utf8
import numpy as np
from scipy.special import gamma as gm
from scipy import stats
import argparse
from pathlib import Path
import csv


def csvread(froot):
    with open(froot, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
    return rows

root = './ass1/files'
root = Path(root)
files = ['hw1q3.csv', 'hw1q4.csv', 'hw1q5.csv']
q3index, q4index, q5index = True, True, True 

if q3index:
    # problem 3
    f1 = root/files[0]
    rows = csvread(f1)
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

    # load functions
    taugm = stats.gamma(a=xi0, scale=1/xi0)
    munorm = stats.norm(mu0, sigma0)    
    u = stats.uniform(0, 1)
    muhat = np.mean(arrY)
    sigmahat2 = np.std(arrY)**2
    cri = sigmahat2**(-n/2)*np.exp(-n/2)
    def cf(mu, sigma2):
        tmp = np.mean((arrY-arrY.mean())**2)/sigma2
        tmp = tmp**(n/2)
        tmp = tmp*np.exp(-np.sum((arrY-mu)**2)/(2*sigma2)+n/2)
        return tmp

    # start simulation
    while len(datalst)<5000:
        mymu = munorm.rvs(size=1)
        mytau = taugm.rvs(size=1)
        myuni = u.rvs(size=1)
        #print(mymu, mytau, myuni)
        if myuni <= cf(mymu, 1/mytau):
            datalst.append((mymu, mytau))
            print(f'the num of data is {len(datalst)}') 
if q4index:
    pass
if q5index:
    pass
