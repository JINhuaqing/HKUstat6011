# coding:utf8
import numpy as np
import prettytable as pt
import pickle
import argparse

parser = argparse.ArgumentParser(description='compute the mean and the varanice of sampling resutls')
parser.add_argument('--file', default='./ass1/savedoc/p5b.pkl', type=str, help='the saved data to be computed')
args = parser.parse_args()
froot = args.file

with open(froot, 'rb') as f:
    data = pickle.load(f)

data = np.array(data)
_, nums = data.shape
means = data.mean(axis=0)
varss = data.var(axis=0)

tb = pt.PrettyTable()
if nums == 4:
    tb.field_names = ['type', 'beta0', 'beta1', 'beta2', 'sigma2']
elif nums == 5:
    tb.field_names = ['type', 'beta0', 'beta1', 'beta2', 'beta3', 'sigma2']
tb.add_row(['means']+list(np.round(means, 4)))
tb.add_row(['variances']+list(np.round(varss, 4)))
print(tb)
