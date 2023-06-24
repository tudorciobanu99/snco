import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from scipy.special import gamma
import math
import time


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
    mu_r = 7.0e5*inv_km_to_eV*(L_an_e/E_avg_an_e - L_an_x/E_avg_an_x)*15e6/1e52*(10/r)**2
    #mu_r = np.sqrt(2)*G_F*(F_nubar_e - F_nubar_x)/(4*np.pi*R**2)*inv_km_to_eV**2
    return mu_r

def mu_r_eff(F_nubar_e, F_nubar_x, r):
    #mu_r_eff = mu_r(F_nubar_e, F_nubar_x, r)*(R**2/(2*r**2))*1/(1 - (R**2)/(2*r**2))
    mu_r_eff = mu_r(F_nubar_e, F_nubar_x, r)*(R**4/(2*r**4)*1/(1 - R**2/(2*r**2)))
    return mu_r_eff

def lambda_r(r):
    t = 0.2
    lambda_r = 7.6e-8*Y_e*rho(r, t)*1e-6
    return lambda_r

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

def rho(r,t):
# r[km],t[s]
#https://arxiv.org/abs/hep-ph/0304056
    r=np.array(r)
    epsilon = 10
    rho_0=1e14*(r**(-2.4)) #g/cm³
    r_s0 = -4.6e3
    v_s = 11.3e3
    a_s = 0.2e3
    r_s = r_s0 + v_s*t + 1/2*a_s*t**2
    rho = 0
    if t <= 1:
        rho = rho_0
    else:
        if r > r_s:
            rho = rho_0
        else:
            f = np.exp((0.28 - 0.69*np.ln(r_s))*np.arcsin(1 - r/r_s))
            rho = rho_0*epsilon*f
    return rho

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
    F_nu_e = F(F_p_nu_e, p)
    F_nu_x = F(F_p_nu_x, p)
    F_nubar_e = F(F_p_nubar_e, p)
    F_nubar_x = F(F_p_nubar_x, p)

    for i in range(len(p)):
        P_p0[i, :] = [0, 0, (F_p_nu_e[i] - F_p_nu_x[i])/(F_nubar_e - F_nubar_x)]
        P_pbar0[i, :] = [0, 0, (F_p_nubar_e[i] - F_p_nubar_x[i])/(F_nubar_e - F_nubar_x)]

    return P_p0, P_pbar0, F_nubar_e, F_nubar_x

def derivs(P, r, p, hierarchy, F_nubar_e, F_nubar_x):
    r = r*inv_km_to_eV
    P_p_nu, P_pbar_nu = np.split(P, 2)
    P_p_nu = P_p_nu.reshape((len(p), 3))
    P_pbar_nu = P_pbar_nu.reshape((len(p), 3))
    P_nu = np.trapz(P_p_nu, p, axis=0)
    P_nubar = np.trapz(P_pbar_nu, p, axis=0)
    rhs_nu = np.zeros((len(p), 3))
    rhs_nubar = np.zeros((len(p), 3))
    for i in range(len(p)):
        rhs_nu[i] = np.cross(omega_p(p[i])*B_array(hierarchy) + lambda_r(r)*L_array() + mu_r_eff(F_nubar_e, F_nubar_x, r)*D_array(P_nu, P_nubar), P_p_nu[i,:])
        rhs_nubar[i] = np.cross(-omega_p(p[i])*B_array(hierarchy) + lambda_r(r)*L_array() + mu_r_eff(F_nubar_e, F_nubar_x, r)*D_array(P_nu, P_nubar), P_pbar_nu[i,:])
    dPdt = np.vstack((rhs_nu, rhs_nubar))
    dPdt = dPdt.flatten()
    return dPdt    

def solve(p, r, r_eval, hierarchy, A, hmax):
    P_p0, P_pbar0, F_nubar_e, F_nubar_x = init(p, A)
    P = np.vstack((P_p0, P_pbar0))
    P = P.flatten()
    packet = 20
    packet_size = int(len(r_eval)/packet)
    for i in range(packet):
        print('Packet ' + str(i+1) + ' out of ' + str(packet) + ' in progress.')
        sol = odeint(derivs, P, r_eval[i*packet_size:(i+1)*packet_size], args = (p, hierarchy, F_nubar_e, F_nubar_x), hmax = hmax)
        P = sol[:,-1]
        print('Packet ' + str(i+1) + ' out of ' + str(packet) + ' completed.')
        P_nu = np.zeros((packet_size, len(p), 3))
        P_nubar = np.zeros((packet_size, len(p), 3))
        for i in range(packet_size):
            P_nu_temp, P_nubar_temp = np.split(np.array(sol[:,i]), 2)
            P_nu[i, :, :] = np.reshape(P_nu_temp, (len(p), 3))
            P_nubar[i, :, :] = np.reshape(P_nubar_temp, (len(p), 3))
        np.save('P_nu_' + str(i+1) + '.csv', P_nu)
        np.save('P_nubar_' + str(i+1) + '.csv', P_nubar)
        print('Packet ' + str(i+1) + ' out of ' + str(packet) + ' saved.')
    #sol = odeint(derivs, P, r_eval, args = (p, hierarchy, F_nubar_e, F_nubar_x), hmax = hmax)

p = np.arange(0.1, 51.1, 1)*1e6
A_nu_e = np.trapz(f(p, E_avg_n_e, alpha), x = p)
A_nu_x = np.trapz(f(p, E_avg_n_x, alpha), x = p)
A_nubar_e = np.trapz(f(p, E_avg_an_e, alpha), x = p)
A_nubar_x = np.trapz(f(p, E_avg_an_x, alpha), x = p)
A = np.array([A_nu_e, A_nu_x, A_nubar_e, A_nubar_x])

r_i = 30/inv_km_to_eV
r_f = 3600/inv_km_to_eV
r = [r_i, r_f]
hmax = (2*np.pi/max(omega_p(p)))/20
r_eval = np.arange(r_i, r_f, hmax)


# P_p0, P_pbar0, F_nubar_e, F_nubar_x = init(p, A)

# plt.figure()
# plt.plot(r_eval*inv_km_to_eV, mu_r_eff(F_nubar_e, F_nubar_x, r_eval*inv_km_to_eV), 'r-')
# plt.plot(r_eval*inv_km_to_eV, [lambda_r(x*inv_km_to_eV) for x in r_eval], 'b-')
# plt.plot(r_eval*inv_km_to_eV, np.repeat(omega_p(p[0]), len(r_eval)), 'g-')
# plt.plot(r_eval*inv_km_to_eV, np.repeat(omega_p(p[-1]), len(r_eval)), 'g-')
# plt.yscale('log')
# plt.show()

hierarchy = 'inverted'
print('Starting solver...')
start_time = time.time()
solve(p, r, r_eval, hierarchy, A, hmax)
print("--- %s seconds ---" % (time.time() - start_time))
# r = sol.t
# sol = sol.y
r = r_eval

# P_nu = np.zeros((len(r), len(p), 3))
# P_nubar = np.zeros((len(r), len(p), 3))
# for i in range(len(r)):
#     P_nu_temp, P_nubar_temp = np.split(np.array(sol[:,i]), 2)
#     P_nu[i, :, :] = np.reshape(P_nu_temp, (len(p), 3))
#     P_nubar[i, :, :] = np.reshape(P_nubar_temp, (len(p), 3))

# rho_p_r_nu = np.zeros((len(r), len(p), 2, 2), dtype = complex)
# rho_p_r_nubar = np.zeros((len(r), len(p), 2, 2), dtype = complex)
# for i in range(len(r)):
#     for j in range(len(p)):
#         J_p_r = 1/2*np.identity(2)*(F_p(p[j], E_avg_n_e, alpha, L_n_e)/A[0] + F_p(p[j], E_avg_n_x, alpha, L_n_x)/A[1]) + 1/2*(P_nu[i,j,0]*sigma_x + P_nu[i,j,1]*sigma_y + P_nu[i,j,2]*sigma_z)*(F_nubar_e - F_nubar_x)
#         J_pbar_r = 1/2*np.identity(2)*(F_p(p[j], E_avg_an_e, alpha, L_an_e)/A[2] + F_p(p[j], E_avg_an_x, alpha, L_an_x)/A[3]) + 1/2*(P_nubar[i,j,0]*sigma_x + P_nubar[i,j,1]*sigma_y + P_nubar[i,j,2]*sigma_z)*(F_nubar_e - F_nubar_x)
#         rho_p_r_nu[i, j, :, :] = (2*np.pi)/(p[j]**2*R**2)*inv_km_to_eV**2*J_p_r
#         rho_p_r_nubar[i, j, :, :] = (2*np.pi)/(p[j]**2*R**2)*inv_km_to_eV**2*J_pbar_r

# Pee = np.zeros((len(r), len(p)))
# Pxx = np.zeros((len(r), len(p)))

# for j in range(len(p)):
#     Pee[:,j] = np.divide(rho_p_r_nu[:,j,0,0], np.trace(rho_p_r_nu[:,j,:,:], axis1=1, axis2=2))
#     Pxx[:,j] = np.divide(rho_p_r_nu[:,j,1,1], np.trace(rho_p_r_nu[:,j,:,:], axis1=1, axis2=2))

# np.savetxt('p_nu.csv', P_nu[-1, :, :], delimiter=',')
# np.savetxt('p_nubar.csv', P_nubar[-1, :, :], delimiter=',')
# np.savetxt('pee.csv', Pee, delimiter=',')
# np.savetxt('pxx.csv', Pxx, delimiter=',')


# Multi-angle simulations
def v_u(r, u):
    v_u_r = np.sqrt(1 - u*(R/r)**2)
    return v_u_r

def init_multi_angle(p, u, A):
    P_p_u0 = np.zeros((len(p), len(u), 3))
    P_pbar_u0 = np.zeros((len(p), len(u), 3))
    F_p_nu_e = np.zeros(len(p))
    F_p_nu_x = np.zeros(len(p))
    F_p_nubar_e = np.zeros(len(p))
    F_p_nubar_x = np.zeros(len(p))
    for i in range(len(p)):
        F_p_nu_e[i] = 2.4*F_p(p[i], E_avg_n_e, alpha, L_n_e)/A[0]
        F_p_nu_x[i] = 1*F_p(p[i], E_avg_n_x, alpha, L_n_x)/A[1]
        F_p_nubar_e[i] = 1.6*F_p(p[i], E_avg_an_e, alpha, L_an_e)/A[2]
        F_p_nubar_x[i] = 1*F_p(p[i], E_avg_an_x, alpha, L_an_x)/A[3]
    F_nubar_e = np.trapz(F(F_p_nubar_e, p), x = u, axis = 1)
    F_nubar_x = np.trapz(F(F_p_nubar_x, p), x = u, axis = 1)

    for i in range(len(p)):
        for j in range(len(u)):
            P_p_u0[i, j, :] = [0, 0, (F_p_nu_e[i] - F_p_nu_x[i])/(F_nubar_e - F_nubar_x)]
            P_pbar_u0[i, j, :] = [0, 0, (F_p_nubar_e[i] - F_p_nubar_x[i])/(F_nubar_e - F_nubar_x)]
    return P_p_u0, P_pbar_u0, F_nubar_e, F_nubar_x

def derivs_multi_angle(P, r, p, u, hierarchy, F_nubar_e, F_nubar_x):
    r = r*inv_km_to_eV
    P_p_u_nu, P_pbar_u_nu = np.split(P, 2)
    P_p_nu = P_p_u_nu.reshape((len(p), 3))
    P_pbar_nu = P_pbar_u_nu.reshape((len(p), 3))
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

def solve_multi_angle(p, r, r_eval, u, hierarchy, A):
    P_p_u0, P_pbar_u0, F_nubar_e, F_nubar_x = init_multi_angle(p, u, A)
    P = np.vstack((P_p_u0, P_pbar_u0))
    P = P.flatten()
    packet = 20
    packet_size = int(len(r_eval)/packet)
    for i in range(packet):
        print('Packet ' + str(i+1) + ' out of ' + str(packet) + ' in progress.')
        sol = odeint(derivs, P, r_eval[i*packet_size:(i+1)*packet_size], args = (p, u, hierarchy, F_nubar_e, F_nubar_x), hmax = hmax)
        P = sol[:,-1]
        print('Packet ' + str(i+1) + ' out of ' + str(packet) + ' completed.')
        P_nu = np.zeros((len(r), len(p), len(u), 3))
        P_nubar = np.zeros((len(r), len(p), len(u), 3))
        for i in range(len(r)):
            P_nu_temp, P_nubar_temp = np.split(np.array(sol[:,i]), 2)
            P_nu[i, :, :] = np.reshape(P_nu_temp, (len(p), 3))
            P_nubar[i, :, :] = np.reshape(P_nubar_temp, (len(p), 3))
        np.savetxt('P_nu_ma_' + str(i+1) + '.csv', P_nu, delimiter = ',')
        np.savetxt('P_nubar_ma_' + str(i+1) + '.csv', P_nubar, delimiter = ',')
        print('Packet ' + str(i+1) + ' out of ' + str(packet) + ' saved.')
    return







    



