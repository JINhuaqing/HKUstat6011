# coding:utf8
import numpy as np
from pathlib import Path
from scipy.stats import uniform, norm, shapiro, multinomial
from scipy.spatial.distance import squareform, pdist
import pickle
import matplotlib.pyplot as plt
import platform

syst = platform.system()
root = Path('./ass3/savedoc')
if syst == 'Linux':
    plt.switch_backend('agg')

np.random.seed(0)


def gen_mixd(k, num, tparas=None):
    if tparas is None:
        tpis = np.arange(1, k+1)
        tpis = tpis/tpis.sum()
        tmus = np.arange(1, k+1)*2*2  
        tsigmas = np.arange(1, k+1)/5 
    else:
        tmus, tsigmas, tpis = tparas
        assert k == len(tmus)
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
        self.loglk = None

    def update(self):
        self.pisk = self.eflagk.mean(axis=1)
        self.musk = (self.eflagk * self.data.reshape(1, -1)).sum(axis=1) \
                    /self.eflagk.sum(axis=1)
        diff2 = (self.data.reshape(1, -1) - self.musk.reshape(-1, 1))**2
        self.sigmask = (self.eflagk * diff2).sum(axis=1) \
                       / self.eflagk.sum(axis=1)
        self.sigmask = np.sqrt(self.sigmask)
        self.loglk = None
    

    def __call__(self, eps=1e-5, output=False):
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
            if output:
                print(f'the {flag:>5}th iteration, the criterion value is {diff:>9.3e}/{eps}')


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



class SSMEM():
    def __init__(self, data, inits, Tsp=0.05, Tme=0.5):
        self.musk = inits[0]
        self.sigmask = inits[1]
        self.pisk = inits[2]
        self.k0 = len(self.musk)
        self.Tsp = Tsp
        self.Tme = Tme
        self.data = data
        self.em = EMk(self.k0, self.data, inits) 
        self.em()

    def _lmerg(self, eflagk):
        mergemat = squareform(1-pdist(eflagk, 'correlation'))
        idx = mergemat.argmax()
        k = eflagk.shape[0]
        mlmerge = mergemat.max()
        print(f'the merge criteria is {mlmerge:.4f}')
        if (mlmerge <= self.Tme) or k == 1:
            return False
        else:
            return idx//k, idx%k

    def _lsplit(self, eflagk, *args):
        idxmax = []
        for ps in eflagk.transpose((1, 0)):
            tmp = multinomial(1, ps).rvs()[0]
            idd, = np.where(tmp==1)
            idxmax.append(int(idd))
        idxmax = np.array(idxmax)
        k = eflagk.shape[0]
        pvalues = []
        for i in range(k):
            d = self.data[idxmax==i]
            _, pv = shapiro(d)
            pvalues.append(pv)
        pvalues = np.array(pvalues)
        mlspl, idx = pvalues.min(), pvalues.argmin() 
        print(f'the split criteria is {mlspl:.4e}')
        if mlspl >= self.Tsp:
            return False
        else:
            return (idx, )

    def _mergerep(self, paras, i, j, value):
        paras = np.delete(paras, [i, j])
        paras = np.insert(paras, i, value)
        return paras

    def _splitrep(self, paras, i, v1, v2):
        paras = np.delete(paras, i)
        paras = np.insert(paras, i, v2)
        paras = np.insert(paras, i, v1)
        return paras


    def __call__(self):
        k = self.k0
        print('-'*100)
        print(f'initial k is {k}')
        em = self.em
        eflagk = em.eflagk
        sigmask = em.sigmask
        musk = em.musk
        pisk = em.pisk
        merge = self._lmerg(eflagk)
        split = self._lsplit(eflagk, musk, sigmask)
        print(f'under this k, we get merge={merge}, split={split}')
        while  merge or split: 
            if merge:
                i, j = merge 
                pinew = pisk[i] + pisk[j]
                munew = (pisk[i]*musk[i] + pisk[j]*musk[j])/pinew
                sigmanew = (pisk[i]*sigmask[i] + pisk[j]*sigmask[j])/pinew
                k -= 1
                init1 = self._mergerep(musk, i, j, munew)
                init2 = self._mergerep(sigmask, i, j, sigmanew)
                init3 = self._mergerep(pisk, i, j, pinew)
                inits = [init1, init2, init3]
                em = EMk(k, self.data, inits) 
                em()

            if (not merge) and split:
                i = split[0]
                pi1 = pi2 = pisk[i]/2
                sigma1 = sigma2 = sigmask[i]
                eps = uniform(0, 1).rvs()
                mu1, mu2 = musk[i]+eps, musk[i]-eps
                k += 1
                init1 = self._splitrep(musk, i, mu1, mu2)
                init2 = self._splitrep(sigmask, i, sigma1, sigma2)
                init3 = self._splitrep(pisk, i, pi1, pi2)
                inits = [init1, init2, init3]
                #em = PEM(k, self.data,inits, [i, i+1], eflagk) 
                em = EMk(k, self.data, inits) 
                em()
                #print(em.musk, em.pisk)

            print('-'*100)
            print(f'now, the k is {k}')
            eflagk = em.eflagk
            sigmask = em.sigmask
            musk = em.musk
            pisk = em.pisk
            merge = self._lmerg(eflagk)
            split = False
            if not merge:
                split = self._lsplit(eflagk, musk, sigmask)
            print(f'under this k, we get merge={merge}, split={split}')
        print('-'*100)
        print('The iteration stops')
        self.em = em





if __name__ == '__main__':
    # True parameters
    # mu       0   -1    1    2
    # sigma    1    2    1  0.5
    # pi    0.25 0.25 0.25 0.25
    with open('./ass3/files/q2.csv') as f:
        data = f.readlines()
    data = [float(i.strip()) for i in data]
    data = np.array(data)
    # data, tmus, tsig, tpi = gen_mixd(4, 1000, [[-1, 2, -3, 4], [1, 2, 1, 0.5], [1/4, 1/4, 1/4, 1/4]])
    k = 1
    # the EM is very sensitive to the initial values, be careful!
    init1 = np.mean(data) + uniform().rvs(k)  
    init2 = np.std(data)  + uniform().rvs(k)
    init3 = np.ones(k)/k
    inits = [init1, init2, init3]
    ssem = SSMEM(data, inits, Tme=0.2, Tsp=0.05)
    ssem()
    em = ssem.em
    print(f'the k is {len(em.musk)}')
    print(em.musk, em.sigmask, em.pisk)



        

            
if __name__ == '__main__':
    k = 4 
    '''
    for k in range(1, 21):
        init1 = np.mean(data) + uniform().rvs(k)  
        init2 = np.std(data)  + uniform().rvs(k)
        init3 = np.ones(k)/k
        em = EMk(k=k, data=data, inits=[init1, init2, init3])
        em(1e-5)
        #print(em.musk, em.sigmask, em.pisk)
        print(f'k={k:>3}  AIC: {em.AIC():>10.3f} BIC: {em.BIC():>10.3f}')

    '''
    init1 = np.array([0, -1, 1, 2])
    init2 = np.array([1, 2, 1, 0.5])
    init3 = np.array([0.25, 0.25, 0.25, 0.25])
    em = EMk(k=4, data=data, inits=[init1, init2, init3])
    #em(1e-5, True)
    #print(em.musk, em.sigmask, em.pisk)
    #print(em.AIC(), em.BIC())

