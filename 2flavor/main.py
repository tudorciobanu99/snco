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
steps = 100
dr = 1e3
r0 = 11*dr
amplitude = flavor_evolution(r0, dr, steps)
np.savetxt('neutrino_e.txt', amplitude[0,:,:])
np.savetxt('antineutrino_e.txt', amplitude[1,:,:])
np.savetxt('neutrino_x.txt', amplitude[2,:,:])
np.savetxt('antineutrino_x.txt', amplitude[3,:,:])

r = np.arange(r0, r0+steps*dr, dr)
# Is = np.zeros(len(r))
# for i in range(len(r)):
#     Is[i] = integrate.simps(amplitude[:,i]*hamiltonians[0,0].f(Es, 0), Es)
amplitude = np.loadtxt('neutrino_e.txt')

plt.ylim(0, 1)
plt.xlim(11, 12)
plt.plot(r*1e-3, amplitude[11,:], 'b-', label='$\\bar{\\nu}_e$')
plt.legend()
plt.xlabel('r (km)')
plt.ylabel('$ P_{\\nu_e \\to \\nu_e}$')
plt.savefig('test.png', dpi=300)
plt.show()
        



