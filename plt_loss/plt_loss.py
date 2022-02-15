# coding:utf-8

import os
import numpy as np
import matplotlib.pyplot as plt

data_path = 'CINICstep' # CIFAR10 CIFAR100 CINICexp CINICstep
files = os.listdir(data_path)

colors = ['r-', 'g-', 'b--', 'm--']

for i, file in enumerate(files):
    file_path = os.path.join(data_path, file)
    results = np.loadtxt(file_path)

    file = file.replace('0.1', '10')
    file = file.replace('0.01', '100')
    file = file.replace('0.02', '50')
    file = file.replace('0.005', '200')
    file_name = file[:-4]

    x = list(range(500))
    y = results[:, 2]
    plt.plot(x, y, colors[i], mec='k', label=file_name, lw=1.5)

plt.legend(loc='upper right',fontsize=15)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.grid(True, ls='--')
plt.show()

