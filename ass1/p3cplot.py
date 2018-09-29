# coding:utf8
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

plt.switch_backend('agg')
plt.figure(figsize=(20, 10))

with open('./ass1/savedoc/p3c.pkl', 'rb') as f:
    data = pickle.load(f)

mus = [i[0] for i in data]
taus = [i[1] for i in data]
plt.subplot(121)
plt.title('the posterior distribution of $\mu$')
plt.hist(mus, bins=30, density=True, color='green')
sns.kdeplot(mus, color='black')
plt.subplot(122)
plt.title(r'the posterior distribution of $\tau$')
plt.hist(taus, bins=30, density=True, color='green')
sns.kdeplot(taus, color='black')
plt.show()
plt.savefig('./ass1/savedoc/p3cfig.jpg')
