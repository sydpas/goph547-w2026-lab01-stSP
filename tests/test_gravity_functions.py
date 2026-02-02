from src.goph547lab01.gravity import(
    gravity_potential_point, 
    gravity_effect_point
)

import numpy as np

def test_gravity_functions():
    x = np.array([1.0, 2.0, 3.0])
    xm = np.array([4.0, 5.0, 6.0])
    m = float(1e7)
    gpp = gravity_potential_point(x, xm, m, G=6.674e-11)
    gep = gravity_effect_point(x, xm, m, G=6.674e-11)
    print(f'Gravity potential point: {gpp}')
    print(f'Gravity effect point: {gep}')


    return

if __name__ == "__main__":
    test_gravity_functions()