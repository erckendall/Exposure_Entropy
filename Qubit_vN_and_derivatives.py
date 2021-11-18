# Plots von Neumann entropy, or its first or second time derivative for given delta value for qubit

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 14})

def von_N(a_sq):
    l_p = (1 + np.sqrt(1 - 4 * (d - d**2 - a_sq * np.exp(-4 * t**2)))) / 2
    l_m = (1 - np.sqrt(1 - 4 * (d - d**2 - a_sq * np.exp(-4 * t**2)))) / 2
    return - (l_p * np.log(l_p) + l_m * np.log(l_m))

def first_derive(a_sq):
    l_p = (1 + np.sqrt(1 - 4 * (d - d**2 - a_sq * np.exp(-4 * t**2)))) / 2
    l_m = (1 - np.sqrt(1 - 4 * (d - d**2 - a_sq * np.exp(-4 * t**2)))) / 2
    l_p_dot = (-8 * a_sq * t * np.exp(- 4 * t**2)) / np.sqrt(1 - 4 * (d - d**2 - a_sq * np.exp(-4 * t**2)))
    l_m_dot = (8 * a_sq * t * np.exp(- 4 * t ** 2)) / np.sqrt(1 - 4 * (d - d ** 2 - a_sq * np.exp(-4 * t ** 2)))
    return - (l_m_dot * (np.log(l_m) + 1) + l_p_dot * (np.log(l_p) + 1))

def sec_derive(a_sq):
    l_p = (1 + np.sqrt(1 - 4 * (d - d**2 - a_sq * np.exp(-4 * t**2)))) / 2
    l_m = (1 - np.sqrt(1 - 4 * (d - d**2 - a_sq * np.exp(-4 * t**2)))) / 2
    l_p_dot = (-8 * a_sq * t * np.exp(- 4 * t**2)) / np.sqrt(1 - 4 * (d - d**2 - a_sq * np.exp(-4 * t**2)))
    l_m_dot = (-8 * a_sq * t * np.exp(- 4 * t ** 2)) / np.sqrt(1 - 4 * (d - d ** 2 - a_sq * np.exp(-4 * t ** 2)))
    l_p_ddot = - ((8 * a_sq * np.exp(-4 * t ** 2) * (1 - 8 * t ** 2)) / np.sqrt(1 - 4 * (d - d ** 2 - a_sq * np.exp(-4 * t ** 2))) + (128 * a_sq ** 2 * t ** 2 * np.exp(-8 * t ** 2)) / ((1 - 4 * (d - d ** 2 - a_sq * np.exp(-4 * t ** 2))))**(3./2.))
    l_m_ddot = ((8 * a_sq * np.exp(-4 * t ** 2) * (1 - 8 * t ** 2)) / np.sqrt(1 - 4 * (d - d ** 2 - a_sq * np.exp(-4 * t ** 2))) + (128 * a_sq ** 2 * t ** 2 * np.exp(-8 * t ** 2)) / ((1 - 4 * (d - d ** 2 - a_sq * np.exp(-4 * t ** 2))))**(3./2.))
    return - ((l_m_ddot * (np.log(l_m) + 1) + l_m_dot**2 / l_m) + (l_p_ddot * (np.log(l_p) + 1) + l_p_dot**2 / l_p))


d = 0.5
for a_sq in np.arange(0., d - d**2 + 0.025, 0.025):
    t = np.arange(0, 1.001, 0.001)
    plt.plot(t, von_N(a_sq), label='{:.2}'.format(a_sq))
    # plt.plot(t, first_derive(a_sq), label='{:.2}'.format(a_sq))
    # plt.plot(t, sec_derive(a_sq), label='{:.2}'.format(a_sq))
plt.legend(frameon=False, title='$|\\alpha|^2$:', ncol=2)
plt.xlim(0,1.)
plt.xlabel('time (arbitrary scale)')
plt.xticks([])
plt.ylabel('$\partial^2_t H_1$') # Change if dot, ddot or just H
plt.show()
