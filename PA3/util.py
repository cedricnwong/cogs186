import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt
import operator
import math
import sys
from datetime import datetime
import os
import glob
import shutil
from PIL import Image
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def plot_particles(positions=None, velocities=None, normalize_velocity=True, alphas=None, name='particle visualization', progresses=None, additonal_info=None):

    # for drawing the the level curves
    X = np.arange(-5, 5, 0.05)
    Y = np.arange(-5, 5, 0.05)
    meshgrid = np.meshgrid(X, Y)
    cmap = cm.colors.LinearSegmentedColormap.from_list('Custom',
                                                       [(0, '#2f9599'),
                                                        (0.45, '#eee'),
                                                           (1, '#8800ff')], N=256)
    # Restating the function

    def f(x, y): return x ** 2 + (y + 1) ** 2 - 5 * \
        np.cos(1.5 * x + 1.5) - 5 * np.cos(2 * y - 1.5)
    try:
        os.mkdir('figures')
    except:
        pass

    def plot_3d_pso(particles=None, velocity=None, normalize=normalize_velocity, ax=None, alpha=None, iteration=None, progress=None, additional_info=None):
        '''
        Helper function to plot the particles in 3D
        '''

        X_grid, Y_grid = meshgrid
        Z_grid = f(X_grid, Y_grid)

        progress = str(progress)[:min(len(str(progress)), 5)]

        # get coordinates and velocity arrays
        if particles is not None:
            X = particles[:, 0]
            Y = particles[:, 1]
            Z = f(X, Y)
            if velocity is not None:
                U, V = velocity.swapaxes(0, 1)
                W = f(X + U, Y + V) - Z

        fig = plt.figure(figsize=(6.8, 6.8))
        ax = plt.axes(projection='3d')

        # Plot the surface.
        surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap=cmap,
                               linewidth=0, antialiased=True, alpha=0.7)
        ax.contour(X_grid, Y_grid, Z_grid, zdir='z',
                   offset=0, levels=30, cmap=cmap)
        if particles is not None:
            if alpha is not None:
                ax.scatter(X, Y, Z, color='#000', depthshade=True,
                           label=f"iter: {iteration}, alpha: {str(alpha)[:min(len(str(alpha)),5)]}, best fitness: {progress}, {additional_info} ")
            else:
                ax.scatter(X, Y, Z, color='#000', depthshade=True,
                           label=f"iter: {iteration}, best fitness: {progress}, {additional_info} ")
            if velocity is not None:
                ax.quiver(X, Y, Z, U, V, W, color='#000',
                          arrow_length_ratio=0., normalize=normalize)

        len_space = 10
        # Customize the axis
        max_z = int((np.max(Z_grid) // len_space + 1)) * len_space
        ax.set_xlim3d(np.min(X_grid), np.max(X_grid))
        ax.set_ylim3d(np.min(Y_grid), np.max(Y_grid))
        ax.set_zlim3d(0, max_z)
        ax.zaxis.set_major_locator(LinearLocator(max_z // len_space + 1))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        ax.set_xlabel('X', fontsize=9)
        ax.set_ylabel('Y', fontsize=9)
        ax.set_zlabel('f(X, Y)', fontsize=9)

        ax.dist = 11

        ax.legend()
        return ax

    def plot_2d_pso(particles=None, velocity=None, normalize=True, ax=None):
        X_grid, Y_grid = meshgrid
        Z_grid = f(X_grid, Y_grid)
        # get coordinates and velocity arrays
        if particles is not None:
            X, Y = particles.swapaxes(0, 1)
            Z = f(X, Y)
            if velocity is not None:
                U, V = velocity.swapaxes(0, 1)
                if normalize:
                    N = np.sqrt(U**2+V**2)
                    U, V = U/N, V/N

        # create new ax if None
        if ax is None:
            fig = plt.figure(figsize=(6.8, 6.8))
            ax = plt.axes()

        # add contours and contours lines
        ax.contour(X_grid, Y_grid, Z_grid, levels=30,
                   linewidths=0.5, colors='#999')
        cntr = ax.contourf(X_grid, Y_grid, Z_grid,
                           levels=30, cmap=cmap, alpha=0.7)
        if particles is not None:
            ax.scatter(X, Y, color='#000')
            if velocity is not None:
                ax.quiver(X, Y, U, V, color='#000',
                          headwidth=2, headlength=2, width=5e-3)

        # add labels and set equal aspect ratio
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xlim(np.min(X_grid), np.max(X_grid))
        ax.set_ylim(np.min(Y_grid), np.max(Y_grid))
        ax.set_aspect(aspect='equal')

    # ploting and saving the graphs
    file_names = []

    # check if alphas are given
    if alphas is not None:
        None
    else:
        alphas = [None]*len(positions)

    for i in range(len(positions)):
        if velocities is not None:
            ax = plot_3d_pso(particles=positions[i], velocity=velocities[i, :, :], normalize=normalize_velocity,
                             progress=progresses[i], ax=None, alpha=alphas[i], iteration=i, additional_info=additonal_info)
        else:
            ax = plot_3d_pso(particles=positions[i], velocity=None, normalize=normalize_velocity,
                             progress=progresses[i], ax=None, alpha=alphas[i], iteration=i, additional_info=additonal_info)
        plt.tight_layout()
        plt.subplots_adjust(top=3)
        plt.savefig('figures/' + name + str(i), dpi=375,
                    bbox_inches='tight', pad_inches=0)
        plt.close()
        file_names.append('figures/' + name + str(i)+'.png')

    # Create the frames
    frames = []
    for i in file_names:
        new_frame = Image.open(i)
        frames.append(new_frame)

    # Save GIF
    frames[0].save(f'{name}.gif', format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=130, loop=0)
    file_names = []
    for i in range(len(positions)):
        if velocities is not None:
            ax = plot_2d_pso(
                particles=positions[i], velocity=velocities[i, :, :], normalize=normalize_velocity, ax=None)
        else:
            ax = plot_2d_pso(
                particles=positions[i], velocity=None, normalize=normalize_velocity, ax=None)
        plt.tight_layout()
        plt.subplots_adjust(top=3)
        plt.savefig('figures/' + name+str(i)+'2d', dpi=375,
                    bbox_inches='tight', pad_inches=0)
        plt.close()
        file_names.append('figures/' + name+str(i)+'2d' + '.png')
        # Create the frames
    frames = []
    for i in file_names:
        new_frame = Image.open(i)
        frames.append(new_frame)

    # Save GIF
    frames[0].save(f'{name} 2d.gif', format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=130, loop=0)
                   
def record_to_pv(particles_record):
     positions = np.zeros((len(particles_record[0]), len(particles_record[0][0]),2))
     velocities = np.zeros((len(particles_record[0]), len(particles_record[0][0]),2))
                    
     for i in range(len(particles_record[0])):
          for p in range(len(particles_record[0][i])):
               positions[i,p,:] = particles_record[0][i][p]
               velocities[i,p,:] = particles_record[1][i][p]
     return positions, velocities