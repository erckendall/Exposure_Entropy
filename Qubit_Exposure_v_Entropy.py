# Plots the n-exposure and the n-Renyi entropy for qubit system in the light-matter interaction

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import matplotlib.path as mpath
from matplotlib.path import Path
import matplotlib
matplotlib.rcParams.update({'font.size': 12})



verts = []
for delta in np.arange(0, 1.01, 0.01):
    verts.append((delta - delta **2, delta))

codes = [Path.LINETO] * 101
codes[0] = Path.MOVETO
path = mpath.Path(verts, codes)
patch = ptch.PathPatch(path, facecolor='none', edgecolor='k')

n = 1.1
bloch_sph = 0 #Change to 1 for Bloch sphere representation


if bloch_sph == 0:
    @np.vectorize
    def ent_exp(a_sq, d):
        l_p = (1 + np.sqrt(1 - 4 * (d - d**2 - a_sq))) / 2
        l_m = (1 - np.sqrt(1 - 4 * (d - d**2 - a_sq))) / 2
        var = 4 * (d - d**2)
        vol = (-4 * a_sq * (l_m ** (n-1) - l_p ** (n-1))) / ((l_p - l_m) * (l_m ** n + l_p ** n))
        # vol = (-4 * a_sq * (np.log(l_m) - np.log(l_p))) / ((l_p - l_m) * (l_m + l_p))
        return var - vol

    @np.vectorize
    def entrop(a_sq, d):
        l_p = (1 + np.sqrt(1 - 4 * (d - d**2 - a_sq))) / 2
        l_m = (1 - np.sqrt(1 - 4 * (d - d**2 - a_sq))) / 2
        entropy = (1/float(1-n)) * np.log(l_p ** n + l_m ** n)
        # entropy = - (l_p ** 2 * np.log(l_p ** 2) + l_m ** 2 * np.log(l_m ** 2))
        return entropy



    x = np.arange(0., 0.25 + 0.001, 0.001)
    y = np.arange(0, 1 + 0.001, 0.001)
    X, Y = np.meshgrid(x,y)
    F = ent_exp(X, Y)
    Z = entrop(X, Y)

    fig, ax = plt.subplots()
    levels2 = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    cs = plt.contourf(X, Y, F, levels=levels2, cmap='summer')
    levels = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    cs2 = plt.contour(X, Y, Z, levels=levels, cmap='gist_heat', linewidths=2)
    ax.add_patch(patch)
    for col in cs.collections:
        col.set_clip_path(patch)
    for col2 in cs2.collections:
        col2.set_clip_path(patch)
    plt.xlabel('|$\\alpha$|$^2$')
    plt.ylabel('$\delta$', rotation=0)
    cbar = plt.colorbar(cs)
    cbar2 = plt.colorbar(cs2)
    cbar.ax.set_title('$E_2$')
    cbar2.ax.set_title('$H_1$')
    plt.show()


elif bloch_sph == 1:

    @np.vectorize
    def bloch(ax, az):
        d = 0.5 * (1+az)
        a_sq = 0.25 * ax**2
        l_p = (1 + np.sqrt(1 - 4 * (d - d ** 2 - a_sq))) / 2
        l_m = (1 - np.sqrt(1 - 4 * (d - d ** 2 - a_sq))) / 2
        var = 4 * (d - d ** 2)
        vol = (-4 * a_sq * (l_m ** (n - 1) - l_p ** (n - 1))) / ((l_p - l_m) * (l_m ** n + l_p ** n))
        return var - vol

    @np.vectorize
    def entropy(ax, az):
        d = 0.5 * (1+az)
        a_sq = 0.25 * ax**2
        l_p = (1 + np.sqrt(1 - 4 * (d - d ** 2 - a_sq))) / 2
        l_m = (1 - np.sqrt(1 - 4 * (d - d ** 2 - a_sq))) / 2
        entr = (1/float(1-n)) * np.log(l_p ** n + l_m ** n)
        # entr = - (l_p ** 2 * np.log(l_p ** 2) + l_m ** 2 * np.log(l_m ** 2))
        return entr

    x = np.arange(0., 1. + 0.001, 0.001)
    y = np.arange(-1., 1. + 0.001, 0.001)
    X, Y = np.meshgrid(x,y)
    F = bloch(X, Y)
    Z = entropy(X, Y)

    fig, ax = plt.subplots(figsize=(8,9))
    levels2 = [-0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    cs = plt.contourf(X, Y, F, levels=levels2, cmap='summer')
    levels = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    cs2 = plt.contour(X, Y, Z, levels=levels, cmap='gist_heat', linewidths=2)
    circ = ptch.Circle((0., 0.), 1., transform=ax.transData)
    for coll in cs.collections:
        coll.set_clip_path(circ)
    for coll2 in cs2.collections:
        coll2.set_clip_path(circ)
    plt.xlabel('$a_x$')
    plt.ylabel('$a_z$', rotation=0)
    cbar = plt.colorbar(cs)
    cbar2 = plt.colorbar(cs2)
    cbar.ax.set_title('$E_n$')
    cbar2.ax.set_title('$H_n$')
    plt.title('{}{}'.format('n=', n))
    plt.show()



