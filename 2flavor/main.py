import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from neutrino import Neutrino
from hamiltonian import Hamiltonian

# Constants
R_nu = 11
G_F = 1.1663787e-23
Y_e = 0.4
deltam = 3e-3
thetav = 0.1
L = 1e51
constants = [deltam, thetav, R_nu, G_F, Y_e, L]

# Initialize particles
Es = np.arange(0.1e6, 31e6, 0.1e6)
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
    densities = np.zeros((4, len(Es)), dtype = object)
    H = np.zeros((4, len(Es)), dtype = object)
    for step in range(steps):
        print(r)
        for i in range(np.size(particles, 0)):
            for j in range(np.size(particles, 1)):
                densities[i,j] = particles[i,j].density_matrix()
                hamiltonians[i,j].reset_hamiltonian(Es[j])

        for i in range(np.size(particles, 0)):
            for j in range(np.size(particles, 1)):
                hamiltonians[i,j].update_hamiltonian(r)
                if r >= R_nu:
                   hamiltonians[i,j].interaction(r, densities, Es)
                particles[i,j].evolve(hamiltonians[i,j].deltaH(dr, particles[0,0].geometrical_factor(R_nu, r)))
                amplitude[i,j,step] = np.abs(particles[i,j].psi[(i % 2)])**2
        print(hamiltonians[0,0].deltaH(dr, particles[0,0].geometrical_factor(R_nu, r)))
        print(amplitude[0,0,step])
        r += dr
    return amplitude

# def flavor_evolution(r0, dr, steps):
#     r = r0
#     amplitude = np.zeros(steps)
#     for step in range(steps):
#         hamiltonian.reset_hamiltonian(particle.E)
#         hamiltonian.update_hamiltonian(r)
#         particle.evolve(hamiltonian.deltaH(dr, particle.geometrical_factor(R_nu, r)))
#         r += dr
#         amplitude[step] = np.abs(particle.psi[0])**2
#     return amplitude

# Evolution
steps = 50
dr = 0.1
r0 = 110*dr
r = np.arange(r0, r0+steps*dr, dr)
amplitude = flavor_evolution(r0, dr, steps)
np.savetxt('nu_e.txt', amplitude[0,:,:])
np.savetxt('bar_nu_e.txt', amplitude[1,:,:])
np.savetxt('nu_x.txt', amplitude[2,:,:])
np.savetxt('bar_nu_x.txt', amplitude[3,:,:])

# plt.plot(r, nu_e[110,:], 'b-', label='$\\nu_e$')
# plt.plot(r, nu_e[0,:], 'k-', label='$\\nu_e$')
# plt.plot(r, nu_e[-1,:], 'r-', label='$\\nu_e$')
plt.plot(r, amplitude[0,110,:], 'k--', label='$\\nu_e$')
plt.ylim(0, 1.1)
plt.legend()
plt.xlabel('r (km)')
plt.ylabel('$ P_{\\nu_e \\to \\nu_e}$')
plt.savefig('test.png', dpi=300)
plt.show()
        



