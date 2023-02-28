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
    return math.sin(2*theta_31), 0, math.cos(2*theta_31)

# L field
def L():
    return 0, 0, 1

def Acc(N_e, E):
    A = 2*math.sqrt(2)*E*G_F*N_e
    return A

def phi(E, E_0, alpha):
    N = ((alpha+1)**(alpha+1))/(E_0*gamma(alpha+1))
    R = N*((E/E_0)**alpha)*math.exp((-1)*(alpha+1)*E/E_0)
    return R
phi_vec = np.vectorize(phi)

def mu(r, mu_opt, mu_0 = 0):
    if mu_opt == 'SN':
        R_0 = 40 # in km
        if r < R_0:
            return mu_0
        return mu_0*(R_0/r)**4 # in eV
    elif mu_opt == 'const':
        return mu_0
    else:
        return 0
mu_vec = np.vectorize(mu)

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
    
def lam(r, option, ne=N_A, t = 1):
    Y_e = 0.5
    m_n = 1.674927471e-24 # in g
    ne_aux = 0
    if option == 'no':
        ne_aux = 0
    elif option == 'const':
        ne_aux = ne
    elif option == 'SN':
        ne_aux = (Y_e/m_n)*rho(r, t)
    else:
        return 0
    return math.sqrt(2)*G_F*inv_cm3_to_eV3*ne_aux # in eV


def structure_constant(n_dim,i,j,k):
    if i==j or i==k or j==k:
        return 0
    if n_dim==3:
        i_next=(i+1)%3
        if  i_next == j:
            return 1
        if i_next == k:
            return -1 
    else:
        print("Dimension not defined")

def cross_prod(A,B):
    n_components=len(A)
    C=[]
    for i in range(n_components):
      sum=0
      for j in range(n_components):
          for k in range(n_components):
            sum=sum+structure_constant(n_components,i,j,k)*A[j]*B[k]
      C.append(sum)
    return C

def read_output(psoln,params):
  n_f,n_dim,n_E= params
  num_diff_nu_compnents=2*n_f*n_dim
  nu,nubar=[],[]
  for l in range(n_dim):
    nu.append([])
    nubar.append([])
    for k in range(n_f):
      nu[l].append([])
      nubar[l].append([])
      for j in range(len(psoln)):
        nu[l][k].append([])
        nubar[l][k].append([])
        for i in range(n_E):
          nu[l][k][j].append(psoln[j][(i*num_diff_nu_compnents)+(l)+(k*2*n_dim)])
          nubar[l][k][j].append(psoln[j][(i*num_diff_nu_compnents)+(l+3)+(k*2*n_dim)])

  return nu, nubar #[Pauli Matrix][Nu_type][time][energy]

def read_two_flavor(nu, nubar):
  nu_e_time,nubar_e_time=[],[]
  nu_x_time,nubar_x_time=[],[]

  for l in range(len(nu[0][0])): #time array length
      nu_e_time.append([])
      nubar_e_time.append([])
      nu_x_time.append([])
      nubar_x_time.append([])
      for i in range(len(nu[0][0][0])): 
        #nu
        P3_x,P3_e=0,0
        if nu[2][0][l][i]>0:
          P3_e=P3_e+nu[2][0][l][i]
        else:
          P3_x=P3_x+nu[2][0][l][i]

        if nu[2][1][l][i]>0:
          P3_e=P3_e+nu[2][1][l][i]
        else:
          P3_x=P3_x+nu[2][1][l][i]

        nu_e_time[l].append(P3_e)
        nu_x_time[l].append(-1*P3_x)

        #nubar
        P3_x,P3_e=0,0
        if nubar[2][0][l][i]>0:
          P3_e=P3_e+nubar[2][0][l][i]
        else:
          P3_x=P3_x+nubar[2][0][l][i]

        if nubar[2][1][l][i]>0:
          P3_e=P3_e+nubar[2][1][l][i]
        else:
          P3_x=P3_x+nubar[2][1][l][i]

        nubar_e_time[l].append(P3_e)
        nubar_x_time[l].append(-1*P3_x)

  return   nu_e_time,nubar_e_time, nu_x_time,nubar_x_time

def read_two_flavor_v1(nu, nubar):
  nu_e_time,nubar_e_time=[],[]
  nu_x_time,nubar_x_time=[],[]

  for l in range(len(nu[0][0])): #time array length
      nu_e_time.append([])
      nubar_e_time.append([])
      nu_x_time.append([])
      nubar_x_time.append([])
    
      for i in range(len(nu[0][0][0])): 
        #nu
        Pee=(1/2)*(1+nu[2][0][l][i]/nu[2][0][0][i])
        Pxx=(1/2)*(1+nu[2][1][l][i]/nu[2][1][0][i])

        nu_e_time[l].append(Pee*nu[2][0][0][i]+(1-Pxx)*(-1)*nu[2][1][0][i])
        nu_x_time[l].append(Pxx*(-1)*nu[2][1][0][i]+(1-Pee)*nu[2][0][0][i])

        #nubar
        Pee=(1/2)*(1+nubar[2][0][l][i]/nubar[2][0][0][i])
        Pxx=(1/2)*(1+nubar[2][1][l][i]/nubar[2][1][0][i])

        nubar_e_time[l].append(Pee*nubar[2][0][0][i]+(1-Pxx)*(-1)*nubar[2][1][0][i])
        nubar_x_time[l].append(Pxx*(-1)*nubar[2][1][0][i]+(1-Pee)*nubar[2][0][0][i])

  return   nu_e_time,nubar_e_time, nu_x_time,nubar_x_time

def read_two_flavor_Probability(nu, nubar):
  Pee_time,nubar_e_time=[],[]
  Peebar_time,nubar_x_time=[],[]

  for l in range(len(nu[0][0])): #time array length
      Pee_time.append([])
      Peebar_time.append([])
    
      for i in range(len(nu[0][0][0])): 
        #nu
        Pee=(1/2)*(1+nu[2][0][l][i]/nu[2][0][0][i])
        Pee_time[l].append(Pee)
        #nubar
        Peebar=(1/2)*(1+nubar[2][0][l][i]/nubar[2][0][0][i])
        Peebar_time[l].append(Peebar)


  return   Pee_time,Peebar_time