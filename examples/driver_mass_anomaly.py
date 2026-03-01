from src.goph547lab01.gravity import(
    gravity_effect_point
)

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sp


def main():
    data = sp.loadmat('../data/anomaly_data.mat')
    x = data['x']
    y = data['y']
    z = data['z']
    rho = data['rho']

    volume = 2 * 2 * 2

    single_cell_mass = volume * rho
    total_cell_mass = np.sum(single_cell_mass)

    x_mass = np.sum(single_cell_mass * x) / total_cell_mass
    y_mass = np.sum(single_cell_mass * y) / total_cell_mass
    z_mass = np.sum(single_cell_mass * z) / total_cell_mass

    print(f'Total mass of anomaly: {total_cell_mass}')
    print(f'Position of Anomaly: {[x_mass, y_mass, z_mass]}')
    print(f'Max cell density: {np.max(rho)}')
    print(f'Mean overall cell density: {np.mean(rho)}')

    # mean densities
    rho_xz = np.mean(rho, axis=0).T
    rho_yz = np.mean(rho, axis=1).T
    rho_xy = np.mean(rho, axis=2).T

    # making coord vectors
    xz = x[0, :, 0]
    yz = y[:, 0, 0]
    xy = z[0, 0, :]

    # making mesh grids
    x_xz, z_xz = np.meshgrid(xz, xy)
    y_yz, z_yz = np.meshgrid(yz, xy)
    x_xy, y_xy = np.meshgrid(yz, xz)

    rhomin, rhomax = np.min(rho), 0.60  # 0.75 from observation

    rho_levels = np.linspace(rhomin, rhomax, 40)

    # plotting cross sections
    fig, ax = plt.subplots(3,1, figsize=(10,14))

    y_xsec = ax[0].contourf(x_xz, z_xz, rho_xz, levels=rho_levels, vmin=rhomin, vmax=rhomax)
    ax[0].plot(x_mass, z_mass, 'xk', markersize=3, label='Barycentre')
    ax[0].set_xlim(-20, 20), ax[0].set_ylim(-15, -5)
    ax[0].set_xlabel('x [m]', fontsize=8), ax[0].set_ylabel('z [m]', fontsize=8)
    ax[0].tick_params(axis='both', labelsize=8)
    ax[0].set_title('Mean Density in xz-plane', fontsize=12, fontweight='bold')
    ax[0].legend()
    cbar=fig.colorbar(y_xsec, ax=ax[0])
    cbar.ax.tick_params(labelsize=8)

    z_xsec = ax[1].contourf(x_xy, y_xy, rho_xy, levels=rho_levels, vmin=rhomin, vmax=rhomax)
    ax[1].plot(x_mass, y_mass, 'xk', markersize=3, label='Barycentre')
    ax[1].set_xlim(-15, 15)
    ax[1].set_ylim(-25, 25)
    ax[1].set_xlabel('x [m]', fontsize=8), ax[0].set_ylabel('y [m]', fontsize=8)
    ax[1].tick_params(axis='both', labelsize=8)
    ax[1].set_title('Mean Density in xy-plane', fontsize=12, fontweight='bold')
    ax[1].legend()
    cbar=fig.colorbar(z_xsec, ax=ax[1])
    cbar.ax.tick_params(labelsize=8)

    x_xsec = ax[2].contourf(y_yz, z_yz, rho_yz, levels=rho_levels, vmin=rhomin, vmax=rhomax)
    ax[2].plot(y_mass, z_mass, 'xk', markersize=3, label='Barycentre')
    ax[2].set_xlim(-15, 15)
    ax[2].set_ylim(-15, -5)
    ax[2].set_xlabel('y [m]', fontsize=8), ax[2].set_ylabel('z [m]', fontsize=8)
    ax[2].tick_params(axis='both', labelsize=8)
    ax[2].set_title('Mean Density in xz-plane', fontsize=12, fontweight='bold')
    ax[2].legend()
    cbar=fig.colorbar(x_xsec, ax=ax[2])
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle('Mean Density Cross Sections with Barycenter', fontweight='bold', fontsize=26)
    plt.savefig('../figures/mass_anom.png', dpi=300)

    plt.show()

    # part 3

    # density slice
    rho_thresh = 0.2 * np.max(rho)

    # create a boolean array (true if density big enough)
    mask = rho >= rho_thresh

    # grabbing true vals
    mean_rho_region = np.mean(rho[mask])

    # and grabbing the corresponding coords
    x_region = x[mask]
    y_region = y[mask]
    z_region = z[mask]

    x_range = (int(x_region.min()), int(x_region.max()))
    y_range = (int(y_region.min()), int(y_region.max()))
    z_range = (int(z_region.min()), int(z_region.max()))

    print(f'x range: {x_range}')
    print(f'y range: {y_range}')
    print(f'z range: {z_range}')
    print(f'Mean density region: {mean_rho_region}')

    # comparing densities
    ratio = mean_rho_region / np.mean(rho)
    print(f'Region mean is {ratio} larger than overall mean')


    # part 4
    zp = [0.0, 100.0]

    x_5, y_5 = np.meshgrid(np.linspace(-100, 100, 40),
                           np.linspace(-100, 100, 40)
                           )

    g_5 = np.zeros((x_5.shape[0], x_5.shape[1], len(zp)))

    for k, z_val in enumerate(zp):
        for i in range(x_5.shape[0]):
            for j in range(x_5.shape[1]):
                x= np.array([x_5[i, j], y_5[i, j], z_val])
                g_5[i,j,k] = gravity_effect_point(x, [x_mass, y_mass, z_mass], total_cell_mass, G=6.674e-11)


    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    for k, z_val in enumerate(zp):
        ax = axes[k]
        g_cf_5 = ax.contourf(x_5, y_5, g_5[:, :, k], levels=40, cmap='viridis_r', vmin=np.min(g_5), vmax=np.max(g_5))
        ax.set_title(f'g at z={z_val} m', fontsize=12, fontweight='bold')
        ax.set_xlabel('x [m]', fontsize=8), ax.set_ylabel('y [m]', fontsize=8)
        ax.tick_params(axis='both', labelsize=8)
        cbar = fig.colorbar(g_cf_5, ax=ax)
        cbar.set_label('g [units]', fontsize=8), cbar.ax.tick_params(labelsize=8)

    fig.suptitle('Survey of 5m for a Forward-Modeled \nGround Based Survey', fontweight='bold', fontsize=18)
    plt.savefig('../figures/5m_forward_model.png', dpi=300)
    plt.show()


    # new data from part 5 -- first order finite diff

    zp_zp = [1.0, 110.0]
    z_0 = zp_zp[0]-0.0
    z_100 = zp_zp[1]-100.0

    g_5_g_5 = np.zeros((x_5.shape[0], x_5.shape[1], len(zp_zp)))  # new grav array

    for k, z_val in enumerate(zp_zp):
        for i in range(x_5.shape[0]):
            for j in range(x_5.shape[1]):
                x= np.array([x_5[i, j], y_5[i, j], z_val])
                g_5_g_5[i,j,k] = gravity_effect_point(x, [x_mass, y_mass, z_mass], total_cell_mass, G=6.674e-11)

    dgzdz_0 = (g_5_g_5[:, :, 0] - g_5[:, :, 0]) / z_0
    dgzdz_100 = (g_5_g_5[:, :, 1] - g_5[:, :, 1]) / z_100

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    ax = axes[0]
    g_cf_5 = ax.contourf(x_5, y_5, dgzdz_0, levels=40, cmap='viridis',
                         vmin=np.min(dgzdz_0), vmax=np.max(dgzdz_0))
    ax.set_title(f'∂gz/∂z at z = 0m', fontsize=12, fontweight='bold')
    ax.set_xlabel('x [m]', fontsize=8), ax.set_ylabel('y [m]', fontsize=8)
    ax.tick_params(axis='both', labelsize=8)
    cbar = fig.colorbar(g_cf_5, ax=ax)
    cbar.set_label('∂gz/∂z', fontsize=8), cbar.ax.tick_params(labelsize=8)

    ax1 = axes[1]
    g_cf_5 = ax1.contourf(x_5, y_5, dgzdz_100, levels=40, cmap='viridis',
                          vmin=np.min(dgzdz_100), vmax=np.max(dgzdz_100))
    ax1.set_title(f'∂gz/∂z at z = 100m', fontsize=12, fontweight='bold')
    ax1.set_xlabel('x [m]', fontsize=8), ax1.set_ylabel('y [m]', fontsize=8)
    ax1.tick_params(axis='both', labelsize=8)
    cbar = fig.colorbar(g_cf_5, ax=ax1)
    cbar.set_label('∂gz/∂z', fontsize=8), cbar.ax.tick_params(labelsize=8)

    # print(f'first order min at 0: {np.min(dgzdz_0)}')
    # print(f'first order max at 0: {np.max(dgzdz_0)}')
    # print(f'first order min at 100: {np.min(dgzdz_100)}')
    # print(f'first order max at 100: {np.max(dgzdz_100)}')

    fig.suptitle('Survey Results using a \nFirst Order Finite Difference', fontweight='bold', fontsize=18)
    plt.savefig('../figures/first_order_finite_difference.png', dpi=300)
    plt.show()


    # part 6

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    ax = axes[0, 0]
    cbar = ax.contourf(x_5, y_5, g_5_g_5[:, :, 0], cmap='viridis_r', vmin=np.min(g_5), vmax=np.max(g_5))
    ax.set_title(f'Elevation of {zp[0]}m', fontsize=12, fontweight='bold')
    ax.set_xlabel('x [m]', fontsize=8), ax.set_ylabel('y [m]', fontsize=8)
    fig.colorbar(cbar, ax=ax)

    ax1 = axes[0, 1]
    cbar = ax1.contourf(x_5, y_5, g_5_g_5[:, :, 1], cmap='viridis_r', vmin=np.min(g_5), vmax=np.max(g_5))
    ax1.set_title(f'Elevation of {zp[1]}m', fontsize=12, fontweight='bold')
    ax1.set_xlabel('x [m]', fontsize=8), ax1.set_ylabel('y [m]', fontsize=8)
    fig.colorbar(cbar, ax=ax1)

    ax2 = axes[1, 0]
    cbar = ax2.contourf(x_5, y_5, g_5_g_5[:, :, 0], cmap='viridis_r', vmin=np.min(g_5_g_5), vmax=np.max(g_5_g_5))
    ax2.set_title(f'Elevation of {zp_zp[0]}m', fontsize=12, fontweight='bold')
    ax2.set_xlabel('x [m]', fontsize=8), ax2.set_ylabel('y [m]', fontsize=8)
    fig.colorbar(cbar, ax=ax2)

    ax3 = axes[1, 1]
    cbar = ax3.contourf(x_5, y_5, g_5_g_5[:, :, 1], cmap='viridis_r', vmin=np.min(g_5_g_5), vmax=np.max(g_5_g_5))
    ax3.set_title(f'Elevation of {zp_zp[1]}m')
    ax3.set_title(f'Elevation of {zp_zp[1]}m', fontsize=12, fontweight='bold')
    ax3.set_xlabel('x [m]', fontsize=8), ax3.set_ylabel('y [m]', fontsize=8)
    fig.colorbar(cbar, ax=ax3)

    fig.suptitle('Vertical Gravity Effect', fontweight='bold', fontsize=26)
    plt.savefig('../figures/vert_grav_eff.png', dpi=300)
    plt.show()


    # part 7 -- second order finite diff

    # np.graident(field, da, axis), using bc more than 2 points like for 1st deriv --> can approx well bc continuous
    #  + not just in one plane (z) but in (x,y)
    d2gzdx2_0 = np.gradient(np.gradient(g_5[:, :, 0], x_5[0, 1] - x_5[0, 0], axis=1),
                            x_5[0, 1] - x_5[0, 0], axis=1)
    d2gzdy2_0 = np.gradient(np.gradient(g_5[:, :, 0], y_5[1, 0] - y_5[0, 0], axis=0),
                            y_5[1, 0] - y_5[0, 0], axis=0)

    d2gzdx2_100 = np.gradient(np.gradient(g_5_g_5[:, :, 1], x_5[0, 1] - x_5[0, 0], axis=1),
                             x_5[0, 1] - x_5[0, 0], axis=1)
    d2gzdy2_100 = np.gradient(np.gradient(g_5_g_5[:, :, 1], y_5[1, 0] - y_5[0, 0], axis=0),
                             y_5[1, 0] - y_5[0, 0], axis=0)

    d2gzdz2_0_lp = -(d2gzdx2_0 + d2gzdy2_0)
    d2gzdz2_100_lp = -(d2gzdx2_100 + d2gzdy2_100)

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    ax = axes[0]
    g_cf_5 = ax.contourf(x_5, y_5, d2gzdz2_0_lp, levels=40, cmap='viridis',
                         vmin=np.min(d2gzdz2_0_lp), vmax=np.max(d2gzdz2_0_lp))
    ax.set_title('∂²gz/∂z² at z = 0m',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('x [m]', fontsize=8), ax.set_ylabel('y [m]', fontsize=8)
    ax.tick_params(axis='both', labelsize=8)
    cbar = fig.colorbar(g_cf_5, ax=ax)
    cbar.set_label('∂²gz/∂z²', fontsize=8), cbar.ax.tick_params(labelsize=8)

    ax1 = axes[1]
    g_cf_5 = ax1.contourf(x_5, y_5, d2gzdz2_100_lp, levels=40, cmap='viridis',
                          vmin=np.min(d2gzdz2_100_lp), vmax=np.max(d2gzdz2_100_lp))
    ax1.set_title('∂²gz/∂z² at z = 100m',
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('x [m]', fontsize=8), ax1.set_ylabel('y [m]', fontsize=8)
    ax1.tick_params(axis='both', labelsize=8)
    cbar = fig.colorbar(g_cf_5, ax=ax1)
    cbar.set_label('∂²gz/∂z²', fontsize=8), cbar.ax.tick_params(labelsize=8)

    # print(f'second order min at 0: {np.min(d2gzdz2_0_lp)}')
    # print(f'second order max at 0: {np.max(d2gzdz2_0_lp)}')
    # print(f'second order min at 100: {np.min(d2gzdz2_100_lp)}')
    # print(f'second order max at 100: {np.max(d2gzdz2_100_lp)}')

    fig.suptitle('Survey Results using a \nSecond Order Finite Difference', fontweight='bold', fontsize=18)
    plt.savefig('../figures/second_order_finite_difference.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()