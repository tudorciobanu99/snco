import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from neutrino import Neutrino
from hamiltonian import Hamiltonian

# Constants
delta_m2 = 3e-3
theta_vac = 0.1
R_nu = 11e3
G_F = 1.1663787e-5
Y_e = 0.4
n_b0 = 0.01246e-3
h_NS = 0.18e3
constants = [delta_m2, theta_vac, R_nu, G_F, Y_e, n_b0, h_NS]

# Neutrino
neutrino = Neutrino(14.95e6, 0, 'e')

# Hamiltonian
hamiltonian = Hamiltonian(constants)
hamiltonian.reset_hamiltonian(neutrino.E)

# Flavor evolution
def flavor_evolution(r0, dr, steps):
    r = r0
    psi = np.zeros((1,2),dtype=np.complex_)
    for i in range(steps):
        hamiltonian.update_hamiltonian(r)
        neutrino.evolve(hamiltonian.deltaH(dr, neutrino.geometrical_factor(R_nu, r)))
        r += dr
        hamiltonian.reset_hamiltonian(neutrino.E)
        psi = np.vstack((psi, neutrino.psi))
    return psi

# Evolution
steps = 100000
dr = 1
r0 = dr
dE = 0.1e6
E0 = dE
# psis = np.zeros((steps, 2, 300),dtype=np.complex_)
# for i in range(300):
#     neutrino = Neutrino(E0+i*dE, 0, 'e')
#     hamiltonian.reset_hamiltonian(neutrino.E)
#     psi = flavor_evolution(r0, dr, steps)
#     psis[:,:,i] = psi[1:,:]

# np.save('psis.npy', psis)

# Energy distribution function
def j(E):
    L = 0.4112e48
    Eavg = 11e6
    T = 2.76e6
    j = 1/18.9686*1/T**3*E**2/(np.exp(E/T-3)+1)
    return j

# Average probability
Es = E0+np.arange(300)*dE
psis = np.load('psis.npy')
I = np.trapz(j(Es)*np.abs(psis[:,0,:])**2, Es)


# Plot
r = np.arange(r0, r0+steps*dr, dr)
# plt.plot(r*1e-3, np.abs(psis[:,0,75])**2, label='E = '+str(Es[75]*1e-6)+'MeV')
# plt.plot(r*1e-3, np.abs(psis[:,0,150])**2, label='E = '+str(Es[150]*1e-6)+'MeV')
# plt.plot(r*1e-3, np.abs(psis[:,0,-1])**2, label='E = '+str(Es[-1]*1e-6)+'MeV')
plt.plot(r*1e-3, I)
plt.xlabel('r (km)')
plt.ylabel('$\\langle P_{\\nu_e \\to \\nu_e}\\rangle$')
plt.xlim(0,100)
plt.ylim(bottom = 0)
#plt.legend()
plt.savefig('probabilities_average.png', dpi=300)
plt.show()
        



