# coding:utf8
from scipy.stats import invgamma, norm, uniform
from scipy.stats import multivariate_normal as mn
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
import platform

# set random seed
np.random.seed(0)
if platform.system() == 'Linux':
    plt.switch_backend('agg')
root = Path('./savedoc')
if platform.system() == 'Linux':
    root = Path('./midterm/savedoc')
if not root.is_dir():
    root.mkdir()

ns = [100, 500]
Js = [2, 5]
beta0, beta1, tau, sigma = 0.2, 1, 0.5, 0.5
sigma_0, sigma_1, alpha1, alpha2, B1, B2 = 10, 10, 0.01, 0.01, 0.01, 0.01


def gendata(n, J, xijs=None, beta0=beta0, beta1=beta1, sigma=sigma, tau=tau):
    if xijs is None:
        xijs = uniform(loc=-1, scale=2).rvs(n*J).reshape(n, J)
    epsijs = norm(loc=0, scale=sigma).rvs(n*J).reshape(n, J)
    bis = norm(loc=0, scale=tau).rvs(n).reshape(n, 1)
    yijs = beta0 + beta1*xijs + bis + epsijs
    return yijs, xijs


def beta0rv(yijs, beta1, xijs, bis, sigma):
    n, J = yijs.shape
    den = sigma**2/sigma_0**2+n*J
    varr = sigma**2/den
    std = np.sqrt(varr)
    mean = (yijs-beta1*xijs-bis).sum()/den
    return norm(loc=mean, scale=std)


def beta1rv(yijs, beta0, xijs, bis, sigma):
    den = sigma**2/sigma_0**2+(xijs**2).sum()
    varr = sigma**2/den
    std = np.sqrt(varr)
    mean = ((yijs-beta0-bis)*xijs).sum()/den
    return norm(loc=mean, scale=std)


def sigma2rv(yijs, beta0, beta1, xijs, bis):
    n, J = yijs.shape
    shape = n*J/2 + alpha1
    scale = 2*B1 + ((yijs-beta0-beta1*xijs-bis)**2).sum()
    scale = scale/2
    return invgamma(a=shape, scale=scale)


def tau2rv(bis):
    n = len(bis)
    shape = n/2 + alpha2
    scale = (bis**2).sum() + 2*B2
    scale = scale/2
    return invgamma(a=shape, scale=scale)


def bisrv(yijs, beta0, beta1, xijs, tau, sigma):
    n, J = yijs.shape
    tau2, sigma2 = tau**2, sigma**2
    varr = sigma2/(sigma2/tau2+J)
    cov = np.diag(np.ones(n)*varr)
    means = (yijs - beta0 - beta1*xijs).sum(axis=1)/(sigma2/tau2+J)
    return mn(mean=means, cov=cov)


def gibbs(num, yijs, xijs, burnin=1000):
    datalst = []
    beta0k, beta1k, tauk, sigmak = 1, 0.5, 0.5, 0.5
    bisk = np.ones(len(xijs))*0.5
    bisk = bisk.reshape(-1, 1)
    flag = 0
    while len(datalst)<num:
        beta0k = beta0rv(yijs, beta1k, xijs, bisk, sigmak).rvs()
        beta1k = beta1rv(yijs, beta0k, xijs, bisk, sigmak).rvs()
        sigma2k = sigma2rv(yijs, beta0k, beta1k, xijs, bisk).rvs()
        sigmak = np.sqrt(sigma2k)
        tau2k = tau2rv(bisk).rvs()
        tauk = np.sqrt(tau2k)
        bisk = bisrv(yijs, beta0k, beta1k, xijs, tauk, sigmak).rvs().reshape(-1, 1)
        flag += 1
        if flag > burnin:
            datalst.append((beta0k, beta1k, tauk, sigmak))
        print(f'iteration {flag}, we have {len(datalst)} samples')
    return datalst


def Pop(data, cutoffs):
    data = np.array(data)
    pops = []
    for cutoff in cutoffs:
        pops.append((data<=cutoff).sum()/len(data))
    return np.array(pops)


def arrtodf(yijs, xijs):
    n, J = yijs.shape
    idx = np.arange(n)
    idx = np.repeat(idx, J)
    yijs, xijs = yijs.reshape(-1), xijs.reshape(-1)
    data = {'Y': yijs, 'X': xijs, 'group': idx}
    return pd.DataFrame(data=data)


def frepvaluetau(data, cutoffs):
    data = np.array(data)
    pops = []
    for cutoff in cutoffs:
        pops.append((data <= cutoff).sum()/len(data))
    return np.array(pops)


def frepvaluebeta(mdf, delta):
    beta1hat = mdf.fe_params["X"]
    beta1hatse = mdf.bse_fe["X"]
    cv = beta1hat - delta
    cv = cv/beta1hatse
    return 1-norm().cdf(cv)


def taubootstrap(df, nums=1000):
    n = len(df)
    idxs = np.random.randint(0, n, size=(nums, n))
    taustats = []
    for ii, idx in enumerate(idxs):
        sample = df.iloc[idx]
        md = smf.mixedlm("Y~1+X", sample, groups=sample["group"])
        mdf = md.fit()
        print(f'we need {nums} times in total, and it is {ii+1}th')
        taustats.append(np.sqrt(mdf.cov_re.iloc[0, 0]))
    return np.array(taustats)


def tauparabootstrap(n, J, parashat, xijs=None, nums=1000):
    taustats = []
    for ii in range(nums):
        yijs, xijs = gendata(n, J, xijs, *parashat)
        sample = arrtodf(yijs, xijs)
        md = smf.mixedlm("Y~1+X", sample, groups=sample["group"])
        mdf = md.fit()
        print(f'we need {nums} times in total, and it is {ii+1}th')
        taustats.append(np.sqrt(mdf.cov_re.iloc[0, 0]))
    return np.array(taustats)


if __name__ == "__main__":
    fig, axes = plt.subplots(nrows=4, ncols=2, sharex='col', figsize=(20, 20))
    for id, n in enumerate(ns):
        for jd, J in enumerate(Js):
            # generate the data
            if (root/f'data{n}_{J}.pkl').is_file():
                with open(root/f'data{n}_{J}.pkl', 'rb') as f:
                    data, xijs, yijs = pickle.load(f)
            else:
                yijs, xijs = gendata(n, J)
                data = None
            datadf = arrtodf(yijs, xijs)
            cutoffs1 = np.linspace(0, 2, 21)
            cutoffs2 = np.linspace(0.05, 1, 20)
            burnin = 5000
            num = 50000 
            numboot = 1000 

            # frequentist method, do the regression
            md = smf.mixedlm("Y~1+X", datadf, groups=datadf["group"])
            mdf = md.fit()
            tauhat, sigmahat = np.sqrt(mdf.cov_re.iloc[0, 0]), np.sqrt(mdf.scale)
            beta0hat, beta1hat = mdf.fe_params['Intercept'], mdf.fe_params['X']
            f1pvalues = frepvaluebeta(mdf, cutoffs1)

            # frequentist method, bootstrap, for tau
            if (root/f'bootstrap{n}_{J}_{numboot}.pkl').is_file():
                with open(root/f'bootstrap{n}_{J}_{numboot}.pkl', 'rb') as f:
                    taubtps = pickle.load(f)
            else:
                taubtps = tauparabootstrap(n, J, [beta0hat, beta1hat, sigmahat, tauhat], xijs, numboot)
            f2pvalues = frepvaluetau(taubtps, cutoffs2)

            # Bayesian method, gibbs sampling
            if data is None:
                data = gibbs(num, yijs, xijs, burnin=burnin)
            dataarr = np.array(data)
            beta1s = dataarr[:, 1]
            taus = dataarr[:, 2]
            b1pvalues = Pop(beta1s, cutoffs1)  # the Pop of beta1
            b2pvalues = Pop(taus, cutoffs2)  # the Pop of tau

            # output
            output1 = "{:10.4f}" * len(b1pvalues)
            output2 = "{:10.4g}" * len(b1pvalues)
            # print(output1.format(*cutoffs))
            # print('Posterior prob of beta1', output2.format(*b1pvalues))
            # print('The p-value of beta1', output2.format(*f1pvalues))

            # plot
            if id*2+jd == 0:
                axes[id*2+jd, 0].set_title(r'$\beta_1$')
            axes[id*2+jd, 0].set_xlabel('delta')
            axes[id*2+jd, 0].set_ylabel(r'pvalue/Pop')
            axes[id*2+jd, 0].plot(cutoffs1, f1pvalues, 'r-o', label=f'n={n}, J={J}, Freq')
            axes[id*2+jd, 0].plot(cutoffs1, b1pvalues, 'g--x', label=f'n={n}, J={J}, Bayes')
            axes[id*2+jd, 0].legend()

            if id*2+jd == 0:
                axes[id*2+jd, 1].set_title(r'$\tau$')
            axes[id*2+jd, 1].set_xlabel('Xi')
            axes[id*2+jd, 1].set_ylabel(r'pvalue/Pop')
            axes[id*2+jd, 1].plot(cutoffs2, f2pvalues, 'r-o', label=f'n={n}, J={J}, Freq')
            axes[id*2+jd, 1].plot(cutoffs2, b2pvalues, 'g--x', label=f'n={n}, J={J}, Bayes')
            axes[id*2+jd, 1].legend()
            # save the data
            if not (root/f'data{n}_{J}.pkl').is_file():
                with open(root/f'data{n}_{J}.pkl', 'wb') as f:
                    pickle.dump([data, xijs, yijs], f)

            if not (root/f'bootstrap{n}_{J}_{numboot}.pkl').is_file():
                with open(root/f'bootstrap{n}_{J}_{numboot}.pkl', 'wb') as f:
                    pickle.dump(taubtps, f)

if platform.system() == 'Linux':
    plt.savefig(root/'plot.jpg')
else:
    plt.show()
