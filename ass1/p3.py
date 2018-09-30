#coding:utf8
import numpy as np
from scipy.special import gamma as gm
from scipy import stats
import argparse
from pathlib import Path
import csv
import pickle


parser = argparse.ArgumentParser(description='python code for problem 3')
parser.add_argument('--is_f', help='whether do part f or not', action='store_true')
args = parser.parse_args()

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
is_f = args.is_f 

# part (c)
# setting
if not is_f: 
    mu0 = 5
    sigma0 = 10
    name1 = 'd'
    name2 = 'e'
    xi0 = 0.01 
else:
    mu0 = 5
    sigma0 = 0.1 
    xi0 = 0.01 
    name1 = 'f1'
    name2 = 'f2'

n = len(arrY)
initnum = 2000
step = 20
size = 50000
datalst = []
ynewlst = []

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
    if flag == initnum: 
        print(f'flag: {flag}, we get the first data')
        datalst.append([muk, tauk])
        rvynew = stats.norm(muk, np.sqrt(1/tauk))
        ynewlst.append(rvynew.rvs())
    if flag > initnum and flag % step == 0:
        print(f'flag: {flag}, we get the  {(flag-initnum)//step+1}th data')
        datalst.append([muk, tauk])
        rvynew = stats.norm(muk, np.sqrt(1/tauk))
        ynewlst.append(rvynew.rvs())

with open(f'./ass1/savedoc/p3{name1}.pkl', 'wb') as f:
    pickle.dump(datalst, f)

with open(f'./ass1/savedoc/p3{name2}.pkl', 'wb') as f:
    pickle.dump(ynewlst, f)
    
