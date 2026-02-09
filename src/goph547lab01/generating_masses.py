import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sp

def masses():
    """Creates 5 point masses.

    Parameters
    ----------

    Returns
    -------
        qwerty
    """
    m = 1.0e7  # [kg]
    masses = np.zeros(5)  # make matrix for masses
    xm = np.zeros((5,3))

    pos = np.array([0,0,-10])  # desired com
    mass_pos= np.zeros(3)  # to store sum(mixi, miyi, mizi) etc

    # generating first 4 masses
    masses[:4] = np.random.normal(m/5, m/100, 4)

    # x coords
    xm[:4,0] = np.random.normal(0, 20, 4)
    # y coords
    xm[:4,1] = np.random.normal(0, 20, 4)
    # z coords
    xm[:4,2] = np.random.normal(-10,2,4)

    masses[4] = m - np.sum(masses[:4])

    # creating mass weighted position of the first 4 masses
    mass_pos[0] = np.sum(masses[:4] * xm[:4,0]) # x coord
    mass_pos[1] = np.sum(masses[:4] * xm[:4,1]) # y coord
    mass_pos[2] = np.sum(masses[:4] * xm[:4,2]) # coord

    xm[4] = (m * pos - mass_pos) / masses[4]

    print(f'mass_pos 0: {mass_pos[0]}')
    print(f'mass_pos 1: {mass_pos[1]}')
    print(f'mass_pos 2: {mass_pos[2]}')
    print(f'xm: {xm}')

    return masses, xm