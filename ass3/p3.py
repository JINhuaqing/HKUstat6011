# coding:utf8
import numpy as np
from scipy.stats import norm, uniform
from pathlib import Path
import pickle
import platform

np.random.seed(0)
root = Path('./ass3/savedoc/')
systt = platform.system()


# True values
beta0, beta1 = 0.5, 1
sigmu, sigeps = 1, 1 
I, J = 100, 5
truepara = np.array([beta0, beta1, sigmu, sigeps])


def gendata(n=I, J=J,  beta0=beta0, beta1=beta1, sigmu=sigmu, sigeps=sigeps):
    xijs = uniform(loc=-1, scale=2).rvs(n*J).reshape(n, J)
    epsijs = norm(loc=0, scale=sigeps).rvs(n*J).reshape(n, J)
    muis = norm(loc=0, scale=sigmu).rvs(n).reshape(n, 1)
    yijs = beta0 + beta1*xijs + muis + epsijs
    return yijs, xijs


class EMlmm():
    def __init__(self, data, init):
        self.yijs, self.xijs = data
        self.I, self.J = self.yijs.shape
        self.beta0k, self.beta1k, self.sigmuk, self.sigepsk = init
        self.muicon1, self.muicon2 = None, None
        self.paras = [self.beta0k, self.beta1k, self.sigmuk, self.sigepsk]

    
    def updateconmu(self):
        yibar = self.yijs.mean(axis=1)
        xibar = self.xijs.mean(axis=1)
        self.muicon1 = self.J*self.sigmuk**2*(yibar-self.beta0k-self.beta1k*xibar) \
                /(self.J*self.sigmuk**2 + self.sigepsk**2)
        self.muicon2 = self.muicon1**2 + (self.sigmuk*self.sigepsk)**2/(self.J*self.sigmuk**2 \
                + self.sigepsk**2)
    
    def updatebetak(self):
        diffyijs = self.yijs - self.muicon1.reshape(-1, 1)
        Y = diffyijs.transpose((1, 0)).reshape(-1)
        Xijs = self.xijs.transpose((1, 0)).reshape(-1)
        X = np.vstack([np.ones(self.I*self.J), Xijs])
        Ymat, Xmat = np.mat(Y), np.mat(X)
        betak = (Xmat*Xmat.T).I*Xmat*Ymat.T
        self.beta0k, self.beta1k = np.array(betak).reshape(-1) 

    def updatesig(self):
        self.sigmuk = np.sqrt(self.muicon2.mean())
        muivar = self.muicon2 - self.muicon1**2
        num = (self.yijs - self.beta0k - self.beta1k*self.xijs-self.muicon1.reshape(-1, 1))**2 + muivar.reshape(-1, 1)
        self.sigepsk = np.sqrt(num.sum()/self.I/self.J)

    def __call__(self, eps=1e-9):
        diff = 1
        while eps < diff:
            lastparas = np.array([self.beta0k, self.beta1k, self.sigmuk, self.sigepsk])
            self.updateconmu()
            self.updatebetak()
            self.updatesig()
            self.paras = np.array([self.beta0k, self.beta1k, self.sigmuk, self.sigepsk])
            diff = np.max(np.abs(self.paras-lastparas))
        

            

    
        
if __name__ == '__main__':
    parass = []
    num = 1000
    for i in range(num): 
        print(f'Iteration {i+1}/{num}')
        yijs, xijs = gendata() 
        em = EMlmm(data=[yijs, xijs], init=[2, 2, 2, 2])
        em()
        parass.append(em.paras)
    parass = np.array(parass)
    parass[:,2:] = parass[:, 2:]**2
    bias = parass.mean(axis=0) - truepara 
    stds = parass.std(axis=0)
    output = '{:<8.4g} ' * 4 
    print('The bias are: ', output.format(*bias))
    print('The stds are: ', output.format(*stds))
