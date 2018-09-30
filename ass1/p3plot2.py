# coding:utf8
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import csv

plt.switch_backend('agg')
plt.figure(figsize=(20, 10))

is_f = True 
if not is_f:
    name = 'e'
else:
    name = 'f2'

with open(f'./ass1/savedoc/p3{name}.pkl', 'rb') as f:
    Ynews = pickle.load(f)

def csvread(froot):
    with open(froot, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
    return rows

Ys = csvread('./ass1/files/hw1q3.csv')
Ys = [float(i[0]) for i in Ys]
plt.title(r'the posterior distribution of Ynew vs origin Y')
plt.hist(Ys, bins=30, density=True, color='red', label='hist for origin Y')
sns.kdeplot(Ys, color='blue', label='density for origin Y')
plt.hist(Ynews, bins=30, density=True, color='green', label='hist for new Y')
sns.kdeplot(Ynews, color='black', label='density for new Y')
plt.legend()

plt.savefig(f'./ass1/savedoc/p3{name}fig.jpg')
