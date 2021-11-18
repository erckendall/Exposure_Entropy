# Plots n-entropy and n-exposure for cross sections at given a_z values for the qutrit system

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
from scipy.linalg import logm
import matplotlib
matplotlib.rcParams.update({'font.size': 12})

omega = 1/3.
upper_bd = 4 * omega ** 2
az = 0.5
rad_2d = np.sqrt(upper_bd - az ** 2)


###################################


# Define the space of operators
# Full unitary dynamics can be represented by three types of transformations:
# Rotations (Sj), 1-axis twisting (Sj^2), 2-axis counter-twisting (Sj^2-Sk^2)
Sx, Sy, Sz = (np.zeros((3,3), dtype=complex) for i in range(3))
Sx2, Sy2, Sz2 = (np.zeros((3,3), dtype=complex) for i in range(3))
Ss = [Sx, Sy, Sz]
sqs = [Sx2, Sy2, Sz2]
for i in range(3):
    sqs[i][np.mod(i+1, 3),np.mod(i+1, 3)] = 1
    sqs[i][np.mod(i+2, 3),np.mod(i+2, 3)] = 1
    Ss[i][np.mod(i + 1, 3), np.mod(i + 2, 3)] = -1j
    Ss[i][np.mod(i + 2, 3), np.mod(i + 1, 3)] = 1j
Ax = np.matmul(Ss[1],Ss[2]) + np.matmul(Ss[2],Ss[1])
Ay = np.matmul(Ss[2],Ss[0]) + np.matmul(Ss[0],Ss[2])
Az = np.matmul(Ss[1],Ss[0]) + np.matmul(Ss[0],Ss[1])


#####################################
@np.vectorize
def Fn(ax, ay):
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = omega
    rho[1, 1] = omega
    rho[2, 2] = omega
    rho[0, 1] = -1j * az / 2
    rho[0, 2] = 1j * ay / 2
    rho[1, 0] = 1j * az / 2
    rho[1, 2] = - 1j * ax / 2
    rho[2, 0] = - 1j * ay / 2
    rho[2, 1] = 1j * ax / 2
    var = np.trace(np.matmul(rho, np.matmul(op, op))) - (np.trace(np.matmul(rho, op))) ** 2
    gamma_n = np.trace(np.linalg.matrix_power(rho, n))
    F_num = np.trace(np.matmul(np.linalg.matrix_power(rho, n - 1), np.matmul((np.matmul(op, rho) - np.matmul(rho, op)), op)))
    # F_num = np.trace(np.matmul(logm(rho), np.matmul((np.matmul(op, rho) - np.matmul(rho, op)), op)))
    return var - (- F_num / gamma_n)


@np.vectorize
def ent(ax, ay):
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = omega
    rho[1, 1] = omega
    rho[2, 2] = omega
    rho[0, 1] = -1j * az / 2
    rho[0, 2] = 1j * ay / 2
    rho[1, 0] = 1j * az / 2
    rho[1, 2] = - 1j * ax / 2
    rho[2, 0] = - 1j * ay / 2
    rho[2, 1] = 1j * ax / 2
    ent = (1/float(1-n)) * np.log(np.trace(np.linalg.matrix_power(rho, n)))
    return ent


# op = sqs[0] - sqs[1] - sqs[2]
# op = 0.5 * (Ax + Az)
op = sqs[1]
n = 2


x = np.arange(0, rad_2d + 0.001, 0.001)
y = np.arange(0, rad_2d + 0.001, 0.001)
X, Y = np.meshgrid(x,y)
F = Fn(X, Y)
E = ent(X, Y)

fig, ax = plt.subplots()
cs = ax.contourf(X,Y,F, 20, cmap=plt.get_cmap('summer'))
cs2 = ax.contour(X,Y,E, 10, cmap=plt.get_cmap('gist_heat'))
circ = ptch.Circle((0., 0.), rad_2d, transform=ax.transData)
for coll in cs.collections:
    coll.set_clip_path(circ)
for coll2 in cs2.collections:
    coll2.set_clip_path(circ)
ax.set_aspect('equal')
# plt.title('{}{}{}{}{}{}'.format('n=', n, ', $a_z=$', az, ', $H=$', '$A_z$'))
ax.text(0.7, 0.8, '$a_z$ = 0.0', transform=ax.transAxes)
plt.xlabel('$a_x$')
plt.ylabel('$a_y$')
plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4])
cbar = plt.colorbar(cs)
cbar2 = plt.colorbar(cs2)
cbar.ax.set_title('$E_{n}$')
cbar2.ax.set_title('$H_{n}$')
plt.title('$S_z$')
plt.show()


