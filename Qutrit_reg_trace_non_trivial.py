# Plots regularised trace term for qutrit system for non-trivial Hamiltonian

import numpy as np
import matplotlib.pyplot as plt
import math



aij = np.ones((3,3))
aij[0,0] = .2
aij[1,0] = aij[0,1] = .1
aij[2,0] = aij[0,2] = .5
aij[2,1] = aij[1,2] = .5
aij[1,1] = .3
aij[2,2] = .5

print(aij)


def Fn(eig_1, eig_2, eps):
    eigs = [eig_0, eig_1, eig_2]
    sm2 = 0
    sm_chk = 0
    for i in range(len(eigs)):
        sm2 += eigs[i] ** (1 + eps)
        for j in range(len(eigs)):
            sm_chk += ((eps + 1) / eps) * eigs[j] ** eps * (eigs[i] - eigs[j]) * np.abs(aij[i,j]) ** 2
            # sm_chk += eigs[j] ** eps * (eigs[i] - eigs[j]) * np.abs(aij[i,j]) ** 2
    return sm_chk / sm2


eps_vals = [.001, .01, .1, .5, 1.]
eig_0s = [0.5]
cnt = 0
for eps in eps_vals:
    cnt += 1
    for eig_0 in eig_0s:
        eig_var = []
        val = []
        var_sc = []
        for eig_1 in np.arange(0., 1 - eig_0 + 0.001, 0.001):
            eig_2 = 1 - eig_0 - eig_1
            eig_var.append(eig_1)
            val.append(Fn(eig_1, eig_2, eps))
        plt.plot(eig_var, val, label='{}{}'.format('$\epsilon$ = ', eps))


plt.title('trace term $\\times$ $(1+\epsilon)/\epsilon$, $\lambda_0$ = 0.5')
plt.xlabel('$\lambda_1$')
plt.ylim(-1., 0.0)
plt.legend(frameon=False)
plt.show()