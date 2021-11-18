# Plots regularised trace term for qutrit system for trivial Hamiltonian

import numpy as np
import matplotlib.pyplot as plt
import math

eig_0 = 0.2
max_k = 10


def Fn(eig_1, eig_2, eps):
    eigs = [eig_0, eig_1, eig_2]
    sm = 0
    sm2 = 0
    sm_chk = 0
    for i in range(len(eigs)):
        sm2 += eigs[i] ** (1 + eps)
        for j in range(len(eigs)):
            sm_chk += ((eps + 1) / eps) * eigs[j] ** eps * (eigs[i] - eigs[j])
            for k in range(max_k):
                sm += np.log(eigs[j]) * (eps * np.log(eigs[j])) ** k * (eigs[i] - eigs[j]) / math.factorial(k + 1)
    # return sm * (1 + eps) / sm2 #### for non-exact calculation: Approaches exact as max_k increased
    return sm_chk / sm2 ##### for exact calculation

eps_vals = [0.01, 0.1, 0.5, 1.0]

# for eps in np.arange(0.501, 0.506, .001):
for eps in eps_vals:
    eig_var = []
    val = []
    for eig_1 in np.arange(0., 1 - eig_0 + 0.001, 0.001):
        eig_2 = 1 - eig_0 - eig_1
        eig_var.append(eig_1)
        val.append(Fn(eig_1, eig_2, eps))
    plt.plot(eig_var, val, label='{}{}'.format('$\epsilon$ = ', eps))


plt.title('{}{}'.format('$\lambda_0$ = ', eig_0))
plt.xlabel('$\lambda_1$')
plt.ylim(-6, 0.1)
plt.legend(frameon=False)
plt.show()