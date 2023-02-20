import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

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
        self.delta_m_12, self.delta_m_13, self.theta_12, self.theta_13, self.theta_23, self.R_nu, self.G_F, self.Y_e = constants

    # Reset Hamiltonian
    #
    # Parameters:
    # E: neutrino energy
    #
    # Returns:
    # None, but updates the Hamiltonian to only vacuum oscillations
    def reset_hamiltonian(self, E):
        self.H = self.delta_m_13/(2*E)*np.array([[np.sin(self.theta_13)**2, 0, np.cos(self.theta_13)*np.sin(self.theta_13)],[0, 0, 0],[np.cos(self.theta_13)*np.sin(self.theta_13), 0, np.cos(self.theta_13)**2]], dtype=np.complex128) + self.delta_m_12/(2*E)*np.array([[np.cos(self.theta_13)**2*np.sin(self.theta_12)**2, np.cos(self.theta_12)*np.cos(self.theta_13)*np.sin(self.theta_12), -np.cos(self.theta_13)*np.sin(self.theta_12)**2*np.sin(self.theta_13)],[np.cos(self.theta_12)*np.cos(self.theta_13)*np.sin(self.theta_12), np.cos(self.theta_12)**2, -np.cos(self.theta_12)*np.sin(self.theta_12)*np.sin(self.theta_13)],[-np.cos(self.theta_13)*np.sin(self.theta_12)**2*np.sin(self.theta_13), -np.cos(self.theta_12)*np.sin(self.theta_12)*np.sin(self.theta_13), np.sin(self.theta_12)**2*np.sin(self.theta_13)**2]], dtype=np.complex_)

    # Update Hamiltonian
    #
    # Parameters:
    # r: current radius
    #
    # Returns:
    # None, but updates the Hamiltonian to include matter effects
    def update_hamiltonian(self, r):
        self.H += 1e-18*np.sqrt(2)*self.G_F*self.n_e(r)*np.array([[1,0,0],[0,0,0],[0,0,0]],dtype = np.complex_)

    # Electron density
    #
    # Parameters:
    # r: current radius
    #
    # Returns:
    # Electron density
    def n_e(self, r):
        n_e = 1.6e36*(5.06)**(-3)*1e-39*1e27*self.Y_e*np.exp((self.R_nu - r)/0.18e3) + 6.0e30*(5.06)**(-3)*1e-39*1e27*self.Y_e*(10e3/r)**3
        #n_e = 1.63e36*(5.06)**(-3)*1e-39*1e27*self.Y_e*np.exp((self.R_nu - r)/0.18e3)
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
        dl = dr*theta*5.076e6
        dH = sp.linalg.expm(-1j*self.H*dl)
        return dH
