import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from neutrino import Neutrino
from hamiltonian import Hamiltonian

# Constants
R_nu = 11e3
G_F = 1.1663787e-5
Y_e = 0.4
deltam = 3e-3
thetam = 0.1
L = 0.4112e48
constants = [deltam, thetam, R_nu, G_F, Y_e, L]

# Initialize particles
Es = np.arange(1e6, 30e6, 1e6)
particles = np.zeros((4, len(Es)), dtype = object)
for i in range(len(Es)):
    particles[0,i] = Neutrino(Es[i], 0, 'e')
    particles[1,i] = Neutrino(Es[i], 0, 'e')
    particles[2,i] = Neutrino(Es[i], 0, 'mu')
    particles[3,i] = Neutrino(Es[i], 0, 'mu')

# Hamiltonian
hamiltonians = np.zeros((4, len(Es)), dtype = object)
for i in range(len(Es)):
    hamiltonians[0,i] = Hamiltonian('neutrino', constants)
    hamiltonians[1,i] = Hamiltonian('antineutrino', constants)
    hamiltonians[2,i] = Hamiltonian('neutrino', constants)
    hamiltonians[3,i] = Hamiltonian('antineutrino', constants)

# Flavor evolution
def flavor_evolution(r0, dr, steps):
    r = r0
    amplitude = np.zeros((4, len(Es), steps))
    for step in range(steps):
        densities = np.zeros((4, len(Es)), dtype = object)
        for i in range(np.size(particles, 0)):
            for j in range(np.size(particles, 1)):
                densities[i,j] = particles[i,j].density_matrix()
                hamiltonians[i,j].reset_hamiltonian(Es[j])

        for i in range(np.size(particles, 0)):
            for j in range(np.size(particles, 1)):
                hamiltonians[i,j].update_hamiltonian(r)
                hamiltonians[i,j].interaction(r, densities, Es)
                particles[i,j].evolve(hamiltonians[i,j].deltaH(dr, particles[0,0].geometrical_factor(R_nu, r)))
                amplitude[i,j,step] = np.abs(particles[i,j].psi[0])**2
        r += dr
    return amplitude

# Evolution
steps = 10000
dr = 0.01e3
r0 = 1100*dr
amplitude = flavor_evolution(r0, dr, steps)
np.save('amplitude.npy', amplitude)

amplitude = np.load('amplitude.npy')

r = np.arange(r0, r0+steps*dr, dr)
plt.plot(r*1e-3, amplitude[0,11,:], label='$\\nu_e$')
plt.ylim(0, 1.1)
plt.xlim(80, 100)
plt.show()


# Plot
# plt.plot(r, np.abs(psi[:,0])**2, label='$\\nu_e$')
# plt.plot(r, np.repeat(np.sin(theta_12)**2*0.9, steps), 'r-')
# # plt.plot(r, hamiltonian.n_e(r), label='$n_e$')
# plt.xlabel('r (km)')
# plt.ylabel('$ P_{\\nu_e \\to \\nu_e}$')
# plt.savefig('prob.png', dpi=300)
# plt.show()
        



