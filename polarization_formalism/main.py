from dependencies import *
from scipy.integrate import odeint

def init(flavor, r_0, r_step, E_0, E_step, weight):
    P0 = [] # initial state
    omega = []

    E_vec = np.arange(E_0, E_0+weight*E_step, E_step)
    
    for i in range(len(E_vec)):
        omega.append(Delta_m2/(2*E_vec[i]*10**6))
        for j in range(2):
            # nu
            P0.append(0)
            P0.append(0)
            P0.append(p_nu)
            # nubar
            P0.append(0)
            P0.append(0)
            P0.append(p_nubar)

