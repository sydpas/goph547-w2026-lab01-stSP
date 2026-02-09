from src.goph547lab01.gravity import(
    gravity_potential_point,
    gravity_effect_point
)

from src.goph547lab01.generating_masses import(
    masses
)

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sp

def main():
    # mass anomaly
    m = 1.0e7  # [kg]
    xm = np.array([0, 0, -10])  # [m]

    # producing 2D grids
    x_25, y_25 = np.meshgrid(np.linspace(-100, 100, 10),
                             np.linspace(-100, 100, 10)
                             )

    x_5, y_5 = np.meshgrid(np.linspace(-100, 100, 40),
                           np.linspace(-100, 100, 40)
                           )

    zp = [0.0, 10.0, 100.0]

    U_25_m1 = np.zeros((x_25.shape[0], x_25.shape[1], len(zp)))  # store grav pot at each grid point for each height
    g_25_m1 = np.zeros((x_25.shape[0], x_25.shape[1], len(zp)))

    U_5_m1 = np.zeros((x_5.shape[0], x_5.shape[1], len(zp)))  # same but for 5
    g_5_m1 = np.zeros((x_5.shape[0], x_5.shape[1], len(zp)))

    U_25_m2 = np.zeros((x_25.shape[0], x_25.shape[1], len(zp)))  # store grav pot at each grid point for each height
    g_25_m2 = np.zeros((x_25.shape[0], x_25.shape[1], len(zp)))

    U_5_m2 = np.zeros((x_5.shape[0], x_5.shape[1], len(zp)))  # same but for 5
    g_5_m2 = np.zeros((x_5.shape[0], x_5.shape[1], len(zp)))

    U_25_m3 = np.zeros((x_25.shape[0], x_25.shape[1], len(zp)))  # store grav pot at each grid point for each height
    g_25_m3 = np.zeros((x_25.shape[0], x_25.shape[1], len(zp)))

    U_5_m3 = np.zeros((x_5.shape[0], x_5.shape[1], len(zp)))  # same but for 5
    g_5_m3 = np.zeros((x_5.shape[0], x_5.shape[1], len(zp)))

    # calling mass generation func
    mass_set_1 = masses()[0]
    mass_set_2 = masses()[0]
    mass_set_3 = masses()[0]
    xm = masses()[1]

    sp.savemat('../data/mass_set_1.mat', {'mass_set_1': mass_set_1})
    sp.savemat('../data/mass_set_2.mat', {'mass_set_2': mass_set_2})
    sp.savemat('../data/mass_set_3.mat', {'mass_set_3': mass_set_3})

    for k, z_val in enumerate(zp):
        for i in range(x_25.shape[0]):
            for j in range(x_25.shape[1]):
                x= np.array([x_25[i, j], y_25[i, j], z_val])
                for mk, xmk in zip(mass_set_1, xm):
                    U_25_m1[i,j,k] += gravity_potential_point(x, xmk, mk, G=6.674e-11)
                    g_25_m1[i,j,k] += gravity_effect_point(x, xmk, mk, G=6.674e-11)

    for k, z_val in enumerate(zp):
        for i in range(x_5.shape[0]):
            for j in range(x_5.shape[1]):
                x= np.array([x_5[i, j], y_5[i, j], z_val])
                for mk, xmk in zip(mass_set_1, xm):
                    U_5_m1[i,j,k] += gravity_potential_point(x, xmk, mk, G=6.674e-11)
                    g_5_m1[i,j,k] += gravity_effect_point(x, xmk, mk, G=6.674e-11)

    for k, z_val in enumerate(zp):
        for i in range(x_25.shape[0]):
            for j in range(x_25.shape[1]):
                x= np.array([x_25[i, j], y_25[i, j], z_val])
                for mk, xmk in zip(mass_set_2, xm):
                    U_25_m2[i,j,k] += gravity_potential_point(x, xmk, mk, G=6.674e-11)
                    g_25_m2[i,j,k] += gravity_effect_point(x, xmk, mk, G=6.674e-11)

    for k, z_val in enumerate(zp):
        for i in range(x_5.shape[0]):
            for j in range(x_5.shape[1]):
                x= np.array([x_5[i, j], y_5[i, j], z_val])
                for mk, xmk in zip(mass_set_2, xm):
                    U_5_m2[i,j,k] += gravity_potential_point(x, xmk, mk, G=6.674e-11)
                    g_5_m2[i,j,k] += gravity_effect_point(x, xmk, mk, G=6.674e-11)

    for k, z_val in enumerate(zp):
        for i in range(x_25.shape[0]):
            for j in range(x_25.shape[1]):
                x= np.array([x_25[i, j], y_25[i, j], z_val])
                for mk, xmk in zip(mass_set_3, xm):
                    U_25_m3[i,j,k] += gravity_potential_point(x, xmk, mk, G=6.674e-11)
                    g_25_m3[i,j,k] += gravity_effect_point(x, xmk, mk, G=6.674e-11)

    for k, z_val in enumerate(zp):
        for i in range(x_5.shape[0]):
            for j in range(x_5.shape[1]):
                x= np.array([x_5[i, j], y_5[i, j], z_val])
                for mk, xmk in zip(mass_set_3, xm):
                    U_5_m3[i,j,k] += gravity_potential_point(x, xmk, mk, G=6.674e-11)
                    g_5_m3[i,j,k] += gravity_effect_point(x, xmk, mk, G=6.674e-11)


    fig, axes = plt.subplots(3, 2, figsize=(12, 16))  # 3 rows, 2 columns
    for k, z_val in enumerate(zp):
        # U plot (left column)
        ax = axes[k, 0]
        U_cf_25 = ax.contourf(x_25, y_25, U_25_m1[:,:,k], levels=40, cmap='viridis', vmin=np.min(U_25_m1), vmax=np.max(U_25_m1))
        ax.plot(x_25, y_25, 'xk', markersize=2)  # overlay grid points
        ax.set_title(f'U at z={z_val} m', fontsize=12, fontweight='bold')
        ax.set_xlabel('x [m]', fontsize=8),ax.set_ylabel('y [m]', fontsize=8)
        ax.tick_params(axis='both', labelsize=8)
        cbar = fig.colorbar(U_cf_25, ax=ax)
        cbar.set_label('g [units]', fontsize=8), cbar.ax.tick_params(labelsize=8)
        # g plot (right column)
        ax1 = axes[k, 1]
        g_cf_25 = ax1.contourf(x_25, y_25, g_25_m1[:,:,k], levels=40, cmap='plasma', vmin=np.min(g_25_m1), vmax=np.max(g_25_m1))
        ax1.plot(x_25, y_25, 'xk', markersize=2)
        ax1.set_title(f'g at z={z_val} m', fontsize=12, fontweight='bold')
        ax1.set_xlabel('x [m]', fontsize=8), ax1.set_ylabel('y [m]', fontsize=8)
        ax1.tick_params(axis='both', labelsize=8)
        cbar = fig.colorbar(g_cf_25, ax=ax1)
        cbar.set_label('g [units]', fontsize=8),cbar.ax.tick_params(labelsize=8)

    fig.suptitle('Survey of 25m for Multi Point Masses with Mass Set 1', fontweight='bold')
    plt.savefig('../figures/multi_mass_set_1_25m.png')

    plt.show()

    fig, axes = plt.subplots(3, 2, figsize=(12, 16))  # 3 rows, 2 columns
    for k, z_val in enumerate(zp):
        # U plot (left column)
        ax = axes[k, 0]
        U_cf_5 = ax.contourf(x_5, y_5, U_5_m1[:,:,k], levels=40, cmap='viridis', vmin=np.min(U_5_m1), vmax=np.max(U_5_m1))
        ax.plot(x_5, y_5, 'xk', markersize=2)  # overlay grid points
        ax.set_title(f'U at z={z_val} m', fontsize=12, fontweight='bold')
        ax.set_xlabel('x [m]', fontsize=8),ax.set_ylabel('y [m]', fontsize=8)
        ax.tick_params(axis='both', labelsize=8)
        cbar = fig.colorbar(U_cf_5, ax=ax)
        cbar.set_label('g [units]', fontsize=8), cbar.ax.tick_params(labelsize=8)
        # g plot (right column)
        ax1 = axes[k, 1]
        g_cf_5 = ax1.contourf(x_5, y_5, g_5_m1[:,:,k], levels=40, cmap='plasma', vmin=np.min(g_5_m1), vmax=np.max(g_5_m1))
        ax1.plot(x_5, y_5, 'xk', markersize=2)
        ax1.set_title(f'g at z={z_val} m', fontsize=12, fontweight='bold')
        ax1.set_xlabel('x [m]', fontsize=8), ax1.set_ylabel('y [m]', fontsize=8)
        ax1.tick_params(axis='both', labelsize=8)
        cbar = fig.colorbar(g_cf_5, ax=ax1)
        cbar.set_label('g [units]', fontsize=8),cbar.ax.tick_params(labelsize=8)

    fig.suptitle('Survey of 5m for Multi Point Masses with Mass Set 1', fontweight='bold')
    plt.savefig('../figures/multi_mass_set_1_5m.png')

    plt.show()

    fig, axes = plt.subplots(3, 2, figsize=(12, 16))  # 3 rows, 2 columns
    for k, z_val in enumerate(zp):
        # U plot (left column)
        ax = axes[k, 0]
        U_cf_25 = ax.contourf(x_25, y_25, U_25_m2[:,:,k], levels=40, cmap='viridis', vmin=np.min(U_25_m2), vmax=np.max(U_25_m2))
        ax.plot(x_25, y_25, 'xk', markersize=2)  # overlay grid points
        ax.set_title(f'U at z={z_val} m', fontsize=12, fontweight='bold')
        ax.set_xlabel('x [m]', fontsize=8),ax.set_ylabel('y [m]', fontsize=8)
        ax.tick_params(axis='both', labelsize=8)
        cbar = fig.colorbar(U_cf_25, ax=ax)
        cbar.set_label('g [units]', fontsize=8), cbar.ax.tick_params(labelsize=8)
        # g plot (right column)
        ax1 = axes[k, 1]
        g_cf_25 = ax1.contourf(x_25, y_25, g_25_m2[:,:,k], levels=40, cmap='plasma', vmin=np.min(g_25_m2), vmax=np.max(g_25_m2))
        ax1.plot(x_25, y_25, 'xk', markersize=2)
        ax1.set_title(f'g at z={z_val} m', fontsize=12, fontweight='bold')
        ax1.set_xlabel('x [m]', fontsize=8), ax1.set_ylabel('y [m]', fontsize=8)
        ax1.tick_params(axis='both', labelsize=8)
        cbar = fig.colorbar(g_cf_25, ax=ax1)
        cbar.set_label('g [units]', fontsize=8),cbar.ax.tick_params(labelsize=8)

    fig.suptitle('Survey of 25m for Multi Point Masses with Mass Set 2', fontweight='bold')
    plt.savefig('../figures/multi_mass_set_2_25m.png')

    plt.show()

    fig, axes = plt.subplots(3, 2, figsize=(12, 16))  # 3 rows, 2 columns
    for k, z_val in enumerate(zp):
        # U plot (left column)
        ax = axes[k, 0]
        U_cf_5 = ax.contourf(x_5, y_5, U_5_m2[:,:,k], levels=40, cmap='viridis', vmin=np.min(U_5_m2), vmax=np.max(U_5_m2))
        ax.plot(x_5, y_5, 'xk', markersize=2)  # overlay grid points
        ax.set_title(f'U at z={z_val} m', fontsize=12, fontweight='bold')
        ax.set_xlabel('x [m]', fontsize=8),ax.set_ylabel('y [m]', fontsize=8)
        ax.tick_params(axis='both', labelsize=8)
        cbar = fig.colorbar(U_cf_5, ax=ax)
        cbar.set_label('g [units]', fontsize=8), cbar.ax.tick_params(labelsize=8)
        # g plot (right column)
        ax1 = axes[k, 1]
        g_cf_5 = ax1.contourf(x_5, y_5, g_5_m2[:,:,k], levels=40, cmap='plasma', vmin=np.min(g_5_m2), vmax=np.max(g_5_m2))
        ax1.plot(x_5, y_5, 'xk', markersize=2)
        ax1.set_title(f'g at z={z_val} m', fontsize=12, fontweight='bold')
        ax1.set_xlabel('x [m]', fontsize=8), ax1.set_ylabel('y [m]', fontsize=8)
        ax1.tick_params(axis='both', labelsize=8)
        cbar = fig.colorbar(g_cf_5, ax=ax1)
        cbar.set_label('g [units]', fontsize=8),cbar.ax.tick_params(labelsize=8)

    fig.suptitle('Survey of 5m for Multi Point Masses with Mass Set 2', fontweight='bold')
    plt.savefig('../figures/multi_mass_set_2_5m.png')

    plt.show()

    fig, axes = plt.subplots(3, 2, figsize=(12, 16))  # 3 rows, 2 columns
    for k, z_val in enumerate(zp):
        # U plot (left column)
        ax = axes[k, 0]
        U_cf_25 = ax.contourf(x_25, y_25, U_25_m3[:,:,k], levels=40, cmap='viridis', vmin=np.min(U_25_m3), vmax=np.max(U_25_m3))
        ax.plot(x_25, y_25, 'xk', markersize=2)  # overlay grid points
        ax.set_title(f'U at z={z_val} m', fontsize=12, fontweight='bold')
        ax.set_xlabel('x [m]', fontsize=8),ax.set_ylabel('y [m]', fontsize=8)
        ax.tick_params(axis='both', labelsize=8)
        cbar = fig.colorbar(U_cf_25, ax=ax)
        cbar.set_label('g [units]', fontsize=8), cbar.ax.tick_params(labelsize=8)
        # g plot (right column)
        ax1 = axes[k, 1]
        g_cf_25 = ax1.contourf(x_25, y_25, g_25_m3[:,:,k], levels=40, cmap='plasma', vmin=np.min(g_25_m3), vmax=np.max(g_25_m3))
        ax1.plot(x_25, y_25, 'xk', markersize=2)
        ax1.set_title(f'g at z={z_val} m', fontsize=12, fontweight='bold')
        ax1.set_xlabel('x [m]', fontsize=8), ax1.set_ylabel('y [m]', fontsize=8)
        ax1.tick_params(axis='both', labelsize=8)
        cbar = fig.colorbar(g_cf_25, ax=ax1)
        cbar.set_label('g [units]', fontsize=8),cbar.ax.tick_params(labelsize=8)

    fig.suptitle('Survey of 25m for Multi Point Masses with Mass Set 3', fontweight='bold')
    plt.savefig('../figures/multi_mass_set_3_25m.png')

    plt.show()

    fig, axes = plt.subplots(3, 2, figsize=(12, 16))  # 3 rows, 2 columns
    for k, z_val in enumerate(zp):
        # U plot (left column)
        ax = axes[k, 0]
        U_cf_5 = ax.contourf(x_5, y_5, U_5_m3[:,:,k], levels=40, cmap='viridis', vmin=np.min(U_5_m3), vmax=np.max(U_5_m3))
        ax.plot(x_5, y_5, 'xk', markersize=2)  # overlay grid points
        ax.set_title(f'U at z={z_val} m', fontsize=12, fontweight='bold')
        ax.set_xlabel('x [m]', fontsize=8),ax.set_ylabel('y [m]', fontsize=8)
        ax.tick_params(axis='both', labelsize=8)
        cbar = fig.colorbar(U_cf_5, ax=ax)
        cbar.set_label('g [units]', fontsize=8), cbar.ax.tick_params(labelsize=8)
        # g plot (right column)
        ax1 = axes[k, 1]
        g_cf_5 = ax1.contourf(x_5, y_5, g_5_m3[:,:,k], levels=40, cmap='plasma', vmin=np.min(g_5_m3), vmax=np.max(g_5_m3))
        ax1.plot(x_5, y_5, 'xk', markersize=2)
        ax1.set_title(f'g at z={z_val} m', fontsize=12, fontweight='bold')
        ax1.set_xlabel('x [m]', fontsize=8), ax1.set_ylabel('y [m]', fontsize=8)
        ax1.tick_params(axis='both', labelsize=8)
        cbar = fig.colorbar(g_cf_5, ax=ax1)
        cbar.set_label('g [units]', fontsize=8),cbar.ax.tick_params(labelsize=8)

    fig.suptitle('Survey of 5m for Multi Point Masses with Mass Set 3', fontweight='bold')
    plt.savefig('../figures/multi_mass_set_3_5m.png')

    plt.show()


if __name__ == '__main__':
    main()
