# Plot exposure as a function of delta for isocurves of entropy

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import matplotlib
matplotlib.rcParams.update({'font.size': 12})


n = 2.

# Define the n-exposure function
def ent_exp(a_sq, d):
    l_p = (1 + np.sqrt(1 - 4 * (d - d**2 - a_sq))) / 2
    l_m = (1 - np.sqrt(1 - 4 * (d - d**2 - a_sq))) / 2
    var = 4 * (d - d**2)
    # vol = (-4 * a_sq * (np.log(l_m) - np.log(l_p))) / ((l_p - l_m) * (l_m + l_p))  ###### for n -> 1
    vol = (-4 * a_sq * (l_m ** (n-1) - l_p ** (n-1))) / ((l_p - l_m) * (l_m ** n + l_p ** n)) ###### for all other n
    return var - vol

def renyi(a_sq):
    l_p = (1 + np.sqrt(1 - 4 * (d - d**2 - a_sq))) / 2
    l_m = (1 - np.sqrt(1 - 4 * (d - d**2 - a_sq))) / 2
    # return - (l_p * np.log(l_p) + l_m * np.log(l_m)) - vN ###### for von Neumann entropy (n -> 1)
    return (1/(1-n)) * np.log(l_m ** n + l_p ** n) - hn ##### for all other n


for hn in np.arange(0.1, 0.6, 0.1):
    a_lst = []
    d_lst = []
    for d in np.arange(0, 1.001, 0.001):
        a_lst.append(opt.fsolve(renyi, 0.)[0])
        d_lst.append(d)
    E_lst = []
    for i in range(len(d_lst)):
        E_lst.append(ent_exp(a_lst[i], d_lst[i]))
    plt.plot(d_lst, E_lst, label='{:.2}'.format(hn))

plt.xlim(0,1.)
plt.legend(frameon=False, title='$H_n$:')
plt.title('{}{}'.format('n = ', n))
plt.ylabel('$E_n$', rotation=0)
plt.xlabel('$\delta$')
plt.show()



