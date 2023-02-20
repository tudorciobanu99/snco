import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from neutrino import Neutrino
from hamiltonian import Hamiltonian

# Constants
R_nu = 11e3
G_F = 1.1663787e-5
Y_e = 0.4
delta_m_12 = 0.753e-4
delta_m_13 = 24.4e-4
theta_12 = 33.4*np.pi/180
theta_13 = 8.6*np.pi/180
theta_23 = 49*np.pi/180
constants = [delta_m_12, delta_m_13, theta_12, theta_13, theta_23, R_nu, G_F, Y_e]

# Neutrino
neutrino = Neutrino(10e6, np.pi/4, 'e')

# Hamiltonian
hamiltonian = Hamiltonian(constants)
hamiltonian.reset_hamiltonian(neutrino.E)

# Flavor evolution
def flavor_evolution(r0, dr, steps):
    r = r0
    psi = np.zeros((steps,3),dtype=np.complex128)
    for i in range(steps):
        hamiltonian.update_hamiltonian(r)
        neutrino.evolve(hamiltonian.deltaH(dr, neutrino.geometrical_factor(R_nu, r)))
        r += dr
        hamiltonian.reset_hamiltonian(neutrino.E)
        psi[i,:] = neutrino.psi
    return psi

# Evolution
steps = 100000
dr = 1e3
r0 = 100*dr
psi = flavor_evolution(r0, dr, steps)

# Energy distribution function
def j(E):
    L = 0.4112e48
    Eavg = 11e6
    T = 2.76e6
    j = 1/18.9686*1/T**3*E**2/(np.exp(E/T-3)+1)
    return j

# Plot
r = np.arange(r0, r0+steps*dr, dr)
plt.plot(r, np.abs(psi[:,0])**2, label='$\\nu_e$')
plt.plot(r, np.repeat(np.sin(theta_12)**2*0.9, steps), 'r-')
# plt.plot(r, hamiltonian.n_e(r), label='$n_e$')
plt.xlabel('r (km)')
plt.ylabel('$ P_{\\nu_e \\to \\nu_e}$')
plt.savefig('prob.png', dpi=300)
plt.show()
        



