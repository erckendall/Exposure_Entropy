# Plots cross section of the n-exposure plane along the alpha^2 axis

import numpy as np
import matplotlib.pyplot as plt


n = 1.5

def ent_exp(a_sq, d):
    l_p = (1 + np.sqrt(1 - 4 * (d - d**2 - a_sq))) / 2
    l_m = (1 - np.sqrt(1 - 4 * (d - d**2 - a_sq))) / 2
    var = 4 * (d - d**2)
    vol = (-4 * a_sq * (l_m ** (n-1) - l_p ** (n-1))) / ((l_p - l_m) * (l_m ** n + l_p ** n))
    # vol = (-4 * a_sq * (np.log(l_m) - np.log(l_p))) / ((l_p - l_m) * (l_m + l_p)) #### for n -> 1
    return var - vol


x = np.arange(0., 0.1-0.1**2 + 0.000001, 0.000001)
plt.plot(x, ent_exp(x, 0.1), label='$\delta = 0.1$')
x = np.arange(0., 0.2-0.2**2 + 0.000001, 0.000001)
plt.plot(x, ent_exp(x, 0.2), label='$\delta = 0.2$')
x = np.arange(0., 0.3-0.3**2 + 0.000001, 0.000001)
plt.plot(x, ent_exp(x, 0.3), label='$\delta = 0.3$')
x = np.arange(0., 0.4-0.4**2 + 0.000001, 0.000001)
plt.plot(x, ent_exp(x, 0.4), label='$\delta = 0.4$')
x = np.arange(0., 0.5-0.5**2 + 0.000001, 0.000001)
plt.plot(x, ent_exp(x, 0.5), label='$\delta = 0.5$')
plt.xlim(0, 0.25)
plt.ylim(0., 1.)
plt.ylabel('$E_n$', rotation=0)
plt.xlabel('|$\\alpha$|$^2$')
plt.legend(frameon=False)
plt.show()



