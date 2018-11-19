# coding:utf8
import numpy as np
from scipy.stats import norm, uniform
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import platform

syst = platform.system()
root = Path('./ass3/savedoc')
if syst == 'Linux':
    plt.switch_backend('agg')

np.random.seed(0)


def gen_mixd(k, num):
    tpis = np.arange(1, k+1)
    tpis = tpis/tpis.sum()
    tmus = np.arange(1, k+1)*2*2  
    tsigmas = np.arange(1, k+1)/5 
    tmps = uniform().rvs(num)
    samples = np.zeros(num) 
    for i in range(k):
       tmpidx = tmps < tpis[i]
       samples[tmpidx] = norm(loc=tmus[i], scale=tsigmas[i]).rvs(tmpidx.sum())
       tmps = tmps - tpis[i]
       tmps[tmpidx] = 1.1
    return samples, tmus, tsigmas, tpis 


class EMk():
    def __init__(self, k, data, inits):
        self.n = len(data)
        self.k = k
        self.data = data
        self.musk = inits[0] 
        self.sigmask = inits[1]
        self.pisk = inits[2]
        self.eflagk = None 
        self.loglk = None

    def conepis(self):
        normmat = np.zeros((self.k, self.n))
        for i in range(self.k):
            muik, sigmaik = self.musk[i], self.sigmask[i]
            normmat[i] = norm(loc=muik, scale=sigmaik).pdf(self.data)
        tmpv = normmat * self.pisk.reshape(-1, 1) 
        self.eflagk = tmpv/tmpv.sum(axis=0)

    def update(self):
        self.pisk = self.eflagk.mean(axis=1)
        self.musk = (self.eflagk * self.data.reshape(1, -1)).sum(axis=1) \
                    /self.eflagk.sum(axis=1)
        diff2 = (self.data.reshape(1, -1) - self.musk.reshape(-1, 1))**2
        self.sigmask = (self.eflagk * diff2).sum(axis=1) \
                       / self.eflagk.sum(axis=1)
        self.sigmask = np.sqrt(self.sigmask)
    

    def __call__(self, eps=1e-5):
        diff = 1
        flag = 0
        while eps < diff:
            flag += 1
            lastmusk = self.musk.copy()
            lastpisk = self.pisk.copy()
            lastsigmask = self.sigmask.copy()
            self.conepis()
            self.update()
            diff1 = np.abs(lastmusk - self.musk).mean()
            diff2 = np.abs(lastpisk - self.pisk).mean()
            diff3 = np.abs(lastsigmask - self.sigmask).mean()
            diff = np.max([diff1, diff2, diff3])
            #print(f'the {flag:>5}th iteration, the criterion value is {diff:>9.3e}/{eps}')


    def loglike(self):
        pdff = norm(loc=self.musk, scale=self.sigmask).pdf
        lgl = 0
        for datum in self.data:
            cpt = self.pisk * pdff(datum)
            lgl += np.log(cpt.sum())
        self.loglk = lgl 

    def AIC(self):
        if self.loglk is None:
            self.loglike()
        return 2*self.k*3-2*self.loglk

    def BIC(self):
        if self.loglk is None:
            self.loglike()
        return np.log(self.n)*3*self.k -2*self.loglk 



            
if __name__ == '__main__':
    # data, tmus, tsigmas, tpis = gen_mixd(k, 1000)
    with open('./ass3/files/q2.csv') as f:
        data = f.readlines()
    data = [float(i.strip()) for i in data]
    data = np.array(data)
    # the EM is very sensitive to the initial values, be careful!
    k = 4 
    for k in range(1, 21):
        init1 = np.mean(data) + uniform().rvs(k)  
        init2 = np.std(data)  + uniform().rvs(k)
        init3 = np.ones(k)/k
        em = EMk(k=k, data=data, inits=[init1, init2, init3])
        em(1e-5)
        print(f'k={k:>3}', f'AIC: {em.AIC():>10.3f}', f'BIC: {em.BIC():>10.3f}')


