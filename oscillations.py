import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import gamma
import math


# Constants
delta_m_sq = 7.5e-5 # eV^2
sin_theta_12_sq = 3.08e-1 # dimensionless
sin_theta_13_sq = 2.18e-2 # dimensionless
sin_theta_23_sq = 4.37e-1 # dimensionless
theta_13 = np.arcsin(np.sqrt(sin_theta_13_sq)) # radians
Delta_m_sq = 2.43e-3 # eV^2
R = 30 # km
L_an_e = 1e52 # erg/s
L_an_x = 1e52 # erg/s
L_n_e = 1e52 # erg/s
L_n_x = 1e52 # erg/s
E_avg_n_e = 11e6 # eV
E_avg_n_x = 25e6 # eV
E_avg_an_e = 16e6 # eV
E_avg_an_x = 25e6 # eV
G_F = 1.166e-23 # eV^-2
Y_e = 0.4 # dimensionless
n_b0 = 1.63e36 # cm^-3
h_NS = 0.8 # km
inv_km_to_eV = 0.197e-9 # eV
inv_cm3_to_eV3 = (0.197)**3*1e-12 # eV3
erg_per_s_to_eV2 = 6.242e11*6.58e-16 # eV2
alpha = 2.3
sigma_x = np.array([[0, 1],[1, 0]])
sigma_y = np.array([[0, -1j],[1j, 0]], dtype = complex)
sigma_z = np.array([[1, 0],[0, -1]])
sigma = np.array([sigma_x, sigma_y, sigma_z])
u = np.sin(np.pi/4)**2

def omega_p(p):
    omega_p = Delta_m_sq/(2*p)
    return omega_p

def B_array(hierarchy):
    B = np.zeros(3)
    if hierarchy == 'normal':
        B = np.array([np.sin(2*theta_13), 0, np.cos(2*theta_13)])
    elif hierarchy == 'inverted':
        B = np.array([np.sin(2*theta_13), 0, -np.cos(2*theta_13)])
    return B

def L_array():
    L = np.array([0, 0, 1])
    return L

def mu_r(F_nubar_e, F_nubar_x, r):
    #mu_r = 7.0e5*inv_km_to_eV*(L_an_e/E_avg_an_e - L_an_x/E_avg_an_x)*15e6/1e52*(10/r)**2
    mu_r = np.sqrt(2)*G_F*(F_nubar_e - F_nubar_x)/(4*np.pi*R**2)*inv_km_to_eV**2
    return mu_r

def mu_r_eff(F_nubar_e, F_nubar_x, r):
    #mu_r_eff = mu_r(F_nubar_e, F_nubar_x, r)*(R**2/(2*r**2))*1/(1 - (R**2)/(2*r**2))
    mu_r_eff = mu_r(F_nubar_e, F_nubar_x, r)*(R**4/(2*r**4)*1/(1 - R**2/(2*r**2)))
    return mu_r_eff

def lambda_r(r):
    n_b = n_b0*inv_cm3_to_eV3*np.exp(-(r - R)/h_NS)
    n_e = Y_e*n_b
    lambda_r = np.sqrt(2)*G_F*n_e
    return lambda_r*0

def f(E, E_avg, alpha):
    f = (E/E_avg)**alpha*np.exp(-(alpha + 1)*E/E_avg)
    return f

def F_p(E, E_avg, alpha, L):
    N = L*erg_per_s_to_eV2/E_avg
    F = N*f(E, E_avg, alpha)
    return F

def F(F_p, p):
    F = np.trapz(F_p, x = p)
    return F

def D_array(P, Pbar):
    D  = P - Pbar
    return D

def P_vec(P_p, p):
    P = np.trapz(P_p, x = p, axis = 0)
    return P

# Initial conditions
def init(p, A):
    P_p0 = np.zeros((len(p), 3))
    P_pbar0 = np.zeros((len(p), 3))
    F_p_nu_e = np.zeros(len(p))
    F_p_nu_x = np.zeros(len(p))
    F_p_nubar_e = np.zeros(len(p))
    F_p_nubar_x = np.zeros(len(p))
    for i in range(len(p)):
        F_p_nu_e[i] = F_p(p[i], E_avg_n_e, alpha, L_n_e)/A[0]
        F_p_nu_x[i] = F_p(p[i], E_avg_n_x, alpha, L_n_x)/A[1]
        F_p_nubar_e[i] = F_p(p[i], E_avg_an_e, alpha, L_an_e)/A[2]
        F_p_nubar_x[i] = F_p(p[i], E_avg_an_x, alpha, L_an_x)/A[3]
    F_nu_e = F(F_p_nu_e*u, p)
    F_nu_x = F(F_p_nu_x*u, p)
    F_nubar_e = F(F_p_nubar_e, p)
    F_nubar_x = F(F_p_nubar_x, p)
    print(F_nu_e/F_nu_x)
    print(F_nubar_e/F_nu_x)

    for i in range(len(p)):
        P_p0[i, :] = [0, 0, (F_p_nu_e[i] - F_p_nu_x[i])/(F_nubar_e - F_nubar_x)]
        P_pbar0[i, :] = [0, 0, (F_p_nubar_e[i] - F_p_nubar_x[i])/(F_nubar_e - F_nubar_x)]

    return P_p0, P_pbar0, F_nubar_e, F_nubar_x

def derivs(r, P, p, hierarchy, F_nubar_e, F_nubar_x):
    r = r*inv_km_to_eV
    print('Progress: r = ' + str(r))
    P_p_nu, P_pbar_nu = np.split(P, 2)
    P_p_nu = P_p_nu.reshape((len(p), 3))
    P_pbar_nu = P_pbar_nu.reshape((len(p), 3))
    rhs_nu = np.zeros((len(p), 3))
    rhs_nubar = np.zeros((len(p), 3))
    for i in range(len(p)):
        rhs_nu[i] = np.cross(omega_p(p[i])*B_array(hierarchy), P_p_nu[i,:])
        rhs_nubar[i] = np.cross(-omega_p(p[i])*B_array(hierarchy), P_pbar_nu[i,:])
    dPdt = np.vstack((rhs_nu, rhs_nubar))
    dPdt = dPdt.flatten()
    return dPdt    

def solve(p, r, r_eval, hierarchy, A):
    P_p0, P_pbar0, F_nubar_e, F_nubar_x = init(p, A)
    P = np.vstack((P_p0, P_pbar0))
    P = P.flatten()
    sol = solve_ivp(derivs, r, P, args = (p, hierarchy, F_nubar_e, F_nubar_x), method="RK45")
    return sol, F_nubar_e, F_nubar_x

p = np.arange(0.1, 51.1, 1)*1e6
A_nu_e = np.trapz(f(p, E_avg_n_e, alpha), x = p)
A_nu_x = np.trapz(f(p, E_avg_n_x, alpha), x = p)
A_nubar_e = np.trapz(f(p, E_avg_an_e, alpha), x = p)
A_nubar_x = np.trapz(f(p, E_avg_an_x, alpha), x = p)
A = np.array([A_nu_e, A_nu_x, A_nubar_e, A_nubar_x])

r_i = 30/inv_km_to_eV
r_f = 100/inv_km_to_eV
r = [r_i, r_f]
r_eval = np.arange(30, 40, 0.1)/inv_km_to_eV
hierarchy = 'inverted'
sol, F_nubar_e, F_nubar_x = solve(p, r, r_eval, hierarchy, A)
r = sol.t
sol = sol.y
P_nu = np.zeros((len(r), len(p), 3))
P_nubar = np.zeros((len(r), len(p), 3))
for i in range(len(r)):
    P_nu_temp, P_nubar_temp = np.split(np.array(sol[:,i]), 2)
    P_nu[i, :, :] = np.reshape(P_nu_temp, (len(p), 3))
    P_nubar[i, :, :] = np.reshape(P_nubar_temp, (len(p), 3))

rho_p_r_nu = np.zeros((len(r), len(p), 2, 2), dtype = complex)
rho_p_r_nubar = np.zeros((len(r), len(p), 2, 2), dtype = complex)
print(len(r))
for i in range(len(r)):
    for j in range(len(p)):
        J_p_r = 1/2*np.identity(2)*(F_p(p[j], E_avg_n_e, alpha, L_n_e)/A[0] + F_p(p[j], E_avg_n_x, alpha, L_n_x)/A[1]) + 1/2*(P_nu[i,j,0]*sigma_x + P_nu[i,j,1]*sigma_y + P_nu[i,j,2]*sigma_z)*(F_nubar_e - F_nubar_x)
        J_pbar_r = 1/2*np.identity(2)*(F_p(p[j], E_avg_an_e, alpha, L_an_e)/A[2] + F_p(p[j], E_avg_an_x, alpha, L_an_x)/A[3]) + 1/2*(P_nubar[i,j,0]*sigma_x + P_nubar[i,j,1]*sigma_y + P_nubar[i,j,2]*sigma_z)*(F_nubar_e - F_nubar_x)
        rho_p_r_nu[i, j, :, :] = (2*np.pi)/(p[j]**2*R**2)*inv_km_to_eV**2*J_p_r
        rho_p_r_nubar[i, j, :, :] = (2*np.pi)/(p[j]**2*R**2)*inv_km_to_eV**2*J_pbar_r

np.savetxt('test.csv', np.array([r,np.abs(rho_p_r_nu[:, 20, 0, 0]), np.abs(rho_p_r_nu[:, 20, 1, 1])]), delimiter = ',')

# Multi-angle simulations
def v_u(r, u):
    v_u_r = np.sqrt(1 - u*(R/r)**2)
    return v_u_r

def init_multi_angle(p, u):
    P_p_u0 = np.zeros((len(p), len(u), 3))
    P_pbar_u0 = np.zeros((len(p), len(u), 3))
    F_p_nu_e = np.zeros(len(p))
    F_p_nu_x = np.zeros(len(p))
    F_p_nubar_e = np.zeros(len(p))
    F_p_nubar_x = np.zeros(len(p))
    for i in range(len(p)):
        F_p_nu_e[i] = 2.4*F_p(p[i], E_avg_n_e, alpha, L_n_e) 
        F_p_nu_x[i] = 1*F_p(p[i], E_avg_n_x, alpha, L_n_x)
        F_p_nubar_e[i] = 1.6*F_p(p[i], E_avg_an_e, alpha, L_an_e)
        F_p_nubar_x[i] = 1*F_p(p[i], E_avg_an_x, alpha, L_an_x)
    F_nubar_e = np.trapz(F(F_p_nubar_e, p), x = u, axis = 1)
    F_nubar_x = np.trapz(F(F_p_nubar_x, p), x = u, axis = 1)

    for i in range(len(p)):
        for j in range(len(u)):
            P_p_u0[i, j, :] = [0, 0, (F_p_nu_e[i] - F_p_nu_x[i])/(F_nubar_e - F_nubar_x)]
            P_pbar_u0[i, j, :] = [0, 0, (F_p_nubar_e[i] - F_p_nubar_x[i])/(F_nubar_e - F_nubar_x)]
    return P_p_u0, P_pbar_u0, F_nubar_e, F_nubar_x

def derivs_multi_angle(P, r, p, u, hierarchy, F_nubar_e, F_nubar_x):
    r = r*inv_km_to_eV
    print(r)
    P_p_u_nu, P_pbar_u_nu = np.split(P, 2)
    #P_p_nu = P_p_u_nu.reshape((len(p), 3))
    #P_pbar_nu = P_pbar_u_nu.reshape((len(p), 3))
    rhs_nu = np.zeros((len(p), len(u), 3))
    rhs_nubar = np.zeros((len(p), len(u), 3))
    for i in range(len(p)):
        for j in range(len(u)):
            inner_int = np.trapz((P_p_u_nu[np.arange(len(p)) != i, np.arange(len(u)) != j, :] - P_pbar_u_nu[np.arange(len(p)) != i, np.arange(len(u)) != j, :])*(1/(v_u(r, u[j]) - v_u(r, u[np.arange(len(u)) != j])) - 1), x = u[np.arange(len(u)) != j], axis = 1)
            outer_int = np.trapz(inner_int, x = p[np.arange(len(p)) != i], axis = 0)
            rhs_nu[i, j] = np.cross(1/v_u(r,u[j])*(omega_p(p[i])*B_array(hierarchy) + lambda_r(r)*L_array()) + mu_r(F_nubar_e, F_nubar_x, r)*outer_int, P_p_u_nu[i,j,:])
            rhs_nubar[i, j] = np.cross(1/v_u(r,u[j])*(-omega_p(p[i])*B_array(hierarchy) + lambda_r(r)*L_array()) + mu_r_eff(F_nubar_e, F_nubar_x, r)*outer_int, P_pbar_u_nu[i,j,:])
    dPdt = np.vstack((rhs_nu, rhs_nubar))
    dPdt = dPdt.flatten()
    return dPdt

def solve_multi_angle(p, u, hierarchy):
    P_p_u0, P_pbar_u0, F_nubar_e, F_nubar_x = init_multi_angle(p, u)
    P = np.vstack((P_p_u0, P_pbar_u0))
    P = P.flatten()
    sol = solve_ivp(derivs_multi_angle, P, r, args = (p, u, hierarchy, F_nubar_e, F_nubar_x))
    return sol







    



