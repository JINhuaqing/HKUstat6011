# coding:utf8
import numpy as np
import matplotlib.pyplot as plt
import platform
from pathlib import Path

systt = platform.system()
root = Path('./ass3/savedoc/')
if systt == 'Linux':
    plt.switch_backend('agg')


E = {}
E['F'] = np.ones(7) * 1/6
E['L'] = np.array([0, 1/12, 1/12, 1/12, 1/6, 1/4, 1/3])

ems = '251326344212463366565535614566523665561326345621443235213263461435421'
ems = [int(i) for i in ems]
ems = [0] + ems

class VA():
    def __init__(self, tm, em=E, ems=ems):
        self.T = tm
        self.E = em
        self.ems = ems
        Pf1 = Pl1 = 0.5
        sf1 = E['F'][ems[1]] * Pf1
        sl1 = E['L'][ems[1]] * Pl1
        self.ftrace = [sf1]
        self.ltrace = [sl1]
        self.fptrace = []
        self.lptrace = []
        self.path = []

    def forward(self):
        for e in self.ems[2:]:
            if self.ftrace[-1]*self.T['F->F'] >=  self.ltrace[-1]*self.T['L->F']:
                self.fptrace.append('F')
            else:
                self.fptrace.append('L')

            if self.ftrace[-1]*self.T['F->L'] >=  self.ltrace[-1]*self.T['L->L']:
                self.lptrace.append('F')
            else:
                self.lptrace.append('L')
            sfi = E['F'][e]*np.max([self.ftrace[-1]*self.T['F->F'], self.ltrace[-1]*self.T['L->F']])
            sli = E['L'][e]*np.max([self.ftrace[-1]*self.T['F->L'], self.ltrace[-1]*self.T['L->L']])
            self.ftrace.append(sfi)
            self.ltrace.append(sli)

    def backward(self):
        if self.ftrace[-1] > self.ltrace[-1]:
            self.path.append('F')
        else:
            self.path.append('L')
        while self.fptrace:
            if self.path[-1] == 'F':
                self.path.append(self.fptrace.pop())
                self.lptrace.pop()
            else:
                self.path.append(self.lptrace.pop())
                self.fptrace.pop()

    def __call__(self):
        self.forward()
        self.backward()
        self.path = list(reversed(self.path))


def path2plot(paths, names=None):
    fig, axess = plt.subplots(nrows=len(paths), figsize=(10, 10))
    if len(paths) == 1:
        axess = [axess]
    if names is None:
        names = np.arange(len(paths))
    for path, axes, name in zip(paths, axess, names):
        x = np.arange(len(path)) 
        y1 = np.zeros(len(path))
        patharr = np.array(path)
        y1[patharr == 'F'] = 1
        y1[patharr == 'L'] = 2
        y2 = y1 - 1
        y21, y22 = y1.copy(), y1.copy()
        y21[y1==1] = y21[y1==1] - 1 
        y22[y1==2] = y22[y1==2] - 1 
        #axes.axis('equal')
        axes.set_title(f'({name})')
        axes.axis('off')
        for idx in x[:-1]:
            label = y1[idx]
            if label == 1:
                axes.text(idx + 0.5, label - 0.5, 'F', color='red', fontsize=10, horizontalalignment='center',
                          verticalalignment='center')
            else:
                axes.text(idx + 0.5, label - 0.5, 'L', color='green', fontsize=10, horizontalalignment='center',
                          verticalalignment='center')
        axes.fill_between(x, y1, y21, step='post', color='green', alpha=0.4)
        axes.fill_between(x, y1, y22, step='post', color='red', alpha=0.4)
        axes.vlines(x, 0, 2) 
        axes.hlines([0, 1, 2], np.min(x), np.max(x))
    return fig, axess
    

T1 = {'F->F': 0.8, 'F->L': 0.2, 'L->F': 0.3, 'L->L': 0.7}
T2 = {'F->F': 0.9, 'F->L': 0.1, 'L->F': 0.15, 'L->L': 0.85}
T3 = {'F->F': 0.95, 'F->L': 0.05, 'L->F': 0.6, 'L->L': 0.4}
T4 = {'F->F': 0.5, 'F->L': 0.5, 'L->F': 0.5, 'L->L': 0.5}

paths = []
for T in [T1, T2, T3, T4]:
    va = VA(tm=T)
    va()
#    print(va.path)
    paths.append(va.path)
path2plot(paths, ['a', 'b', 'c', 'd'])
plt.show()
plt.savefig(root/'p4.jpg')
    



