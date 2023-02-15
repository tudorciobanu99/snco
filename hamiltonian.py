import numpy as np
import matplotlib.pyplot as plt

class Hamiltonian:
    # Initialization
    #
    # Parameters:
    # constants: list of constants
    #
    # Returns:
    # Hamiltonian object with the following attributes:
    # delta_m2: mass splitting
    # theta_vac: vacuum mixing angle
    # R_nu: radius of the neutrino sphere
    # G_F: Fermi constant
    # Y_e: electron fraction
    # n_b0: baryon density at the center of the star
    # h_NS: scale height of the neutron star
    def __init__(self, constants):
        self.delta_m2, self.theta_vac, self.R_nu, self.G_F, self.Y_e, self.n_b0, self.h_NS = constants

    # Reset Hamiltonian
    #
    # Parameters:
    # E: neutrino energy
    #
    # Returns:
    # None, but updates the Hamiltonian to only vacuum oscillations
    def reset_hamiltonian(self, E):
        self.H = self.delta_m2/(4*E)*np.array([[-np.cos(2*self.theta_vac), np.sin(2*self.theta_vac)],[np.sin(2*self.theta_vac), np.cos(2*self.theta_vac)]],dtype=np.complex_)

    # Update Hamiltonian
    #
    # Parameters:
    # r: current radius
    #
    # Returns:
    # None, but updates the Hamiltonian to include matter effects
    def update_hamiltonian(self, r):
        self.H += 1e9*np.sqrt(2)*self.G_F*self.Y_e*self.n_e(r)*np.array([[1,0],[0,0]],dtype=np.complex_)

    # Electron density
    #
    # Parameters:
    # r: current radius
    #
    # Returns:
    # Electron density
    def n_e(self, r):
        n_e = self.n_b0*np.exp((self.R_nu-r)/self.h_NS)
        return n_e
    
    # Hamiltonian change
    #
    # Parameters:
    # theta_0: initial angle
    # dr: step size in radius
    # theta: step size in angle
    #
    # Returns:
    # Hamiltonian change for the given step size
    def deltaH(self, dr, theta):
        lam = np.sqrt(self.H[0,0]**2 + np.abs(self.H[0,1])**2)
        dl = dr*theta*5.076e6
        dH = 1/lam*np.array([[lam*np.cos(lam*dl) - 1j*self.H[0,0]*np.sin(lam*dl), -1j*self.H[0,1]*np.sin(lam*dl)],[-1j*np.conj(self.H[0,1])*np.sin(lam*dl), lam*np.cos(lam*dl) + 1j*self.H[0,0]*np.sin(lam*dl)]])
        return dH
