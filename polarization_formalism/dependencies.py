import numpy as np
import math
from scipy.special import gamma

# Constants
G_F = 1.1663787e-23 # Fermi constant in eV^-2
Delta_m2 = 2.43e-3 # in eV^2
delta_m2 = 7.54e-5 # in eV^2
theta_31 = np.arcsin(math.sqrt(2.34-2)) # in radians
theta_21 = np.arcsin(math.sqrt(3.08e-1))# in radians
N_A = 6.022140857e23 # Avogadro's number
h_bar = 6.582119514e-16 # in eV*s
c = 2.99792458e8 # in m/s
eV_to_inv_m = 1/(h_bar*c) # in 1/m
eV_to_inv_km = 1e3*eV_to_inv_m # in 1/km
inv_cm3_to_eV3 = (1.973e-5)**3 # in eV^3
erg_to_MeV = 6.2415e5 # in MeV
R_nu = 11 # in km

# B field
def B():
    return np.array(math.sin(2*theta_31), 0, math.cos(2*theta_31))

def L():
    return np.array(0, 0, 1)

def omega_E(E):
    omega = Delta_m2/(2*E)
    return omega

def mu(r, kind, mu_0 = 0):
    if kind == 'SN':
        R_0 = 40 # in km
        if r < R_0:
            return mu_0
        return mu_0*(R_0/r)**4 # in eV
    elif kind == 'const':
        return mu_0
    else:
        return 0
    

def rho(r, t):
    r = np.array(r) # in km
    rho_0 = 10e14*(r**(-2.4)) #in g/cm^3
    if t < 1:
        return rho_0
    else:
        eps = 10
        r_s0 = -4.6e3 # in km
        v_s = 11.3e3 # in km/s
        a_s = 0.2e3 # in km/s^2
        r_s = r_s0 + v_s*a_s*t**2 # in km
        rho = []
        for i in range(len(r)):
            if r[i] <= r_s:
                aux = (0.28 - 0.69*np.log(r[i]))*(np.arcsin(1- r[i]/r_s)**1.1)
                f = np.exp(aux)
                rho.append(eps*f*rho_0[i])
            else:
                rho.append(rho_0[i])
        return np.array(rho)
    
def lam(r, kind, ne=N_A, t = 1):
    Y_e = 0.5
    m_n = 1.674927471e-24 # in g
    ne_aux = 0
    if kind == 'default':
        ne_aux = 0
    elif kind == 'const':
        ne_aux = ne
    elif kind == 'SN':
        ne_aux = (Y_e/m_n)*rho(r, t)
    else:
        return 0
    return math.sqrt(2)*G_F*inv_cm3_to_eV3*ne_aux # in eV