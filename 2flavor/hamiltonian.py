import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import cmath

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
    def __init__(self, kind, constants):
        self.deltam, self.thetam, self.R_nu, self.G_F, self.Y_e, self.L = constants
        self.kind = kind

    # Reset Hamiltonian
    #
    # Parameters:
    # E: neutrino energy
    #
    # Returns:
    # None, but updates the Hamiltonian to only vacuum oscillations
    def reset_hamiltonian(self, E):
        self.H = self.deltam/(4*E)*np.array([[-np.cos(2*self.thetam), np.sin(2*self.thetam)],[np.sin(2*self.thetam), np.cos(2*self.thetam)]], dtype = np.complex_)

    # Update Hamiltonian
    #
    # Parameters:
    # r: current radius
    #
    # Returns:
    # None, but updates the Hamiltonian to include matter effects
    def update_hamiltonian(self, r):
        if self.kind == 'neutrino':
            self.H += 1/2*1e-18*np.sqrt(2)*self.G_F*self.n_e(r)*np.array([[1,0],[0,-1]],dtype = np.complex_)
        else:
            self.H -= 1/2*1e-18*np.sqrt(2)*self.G_F*self.n_e(r)*np.array([[1,0],[0,-1]],dtype = np.complex_)


    # Electron density
    #
    # Parameters:
    # r: current radius
    #
    # Returns:
    # Electron density
    def n_e(self, r):
        n_e = 1.6e36*0.196**3*1e-12*self.Y_e*np.exp((self.R_nu - r)/0.18e3) + 6.0e30*0.197**3*1e-32*self.Y_e*(10e3/r)**3
        return n_e

    def interaction(self, r, particles, Es):
        Eavg = [11e6, 16e6, 25e6]
        Te = sp.integrate.simps((particles[0,:]*self.f(Es, 0)*self.L/Eavg[0] - np.conjugate(particles[1,:])*self.f(Es, 1)*self.L/Eavg[1]),Es)
        Tx = sp.integrate.simps((particles[2,:]*self.f(Es, 2)*self.L/Eavg[2] - np.conjugate(particles[3,:])*self.f(Es, 2)*self.L/Eavg[2]),Es)
        H_int = np.sqrt(2)*self.G_F*1e-18/(2*np.pi*self.R_nu**2)*self.geometric_factor(r)*(Te + Tx)
        if self.kind == 'neutrino':
            self.H += H_int
        else:
            self.H += np.array([[-H_int[0,0],-H_int[1,0]],[-H_int[0,1],-H_int[1,1]]], dtype = np.complex_)
    
    def geometric_factor(self, r):
        D = 1/2*(1 - cmath.sqrt(1 - (self.R_nu/r)**2))**2
        return D

    def f(self, E, type):
        T = [2.76e6, 4.01e6, 6.26e6]
        f = 1/18.9686*1/T[type]**3*E**2/(np.exp(E/T[type]-3)+1)
        return f

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
        lam = np.sqrt(self.H[0,0]**2 + np.abs(self.H[0,1])**2)
        dH = 1/lam * np.array([[lam*np.cos(lam*dl) - 1j*self.H[0,0]*np.sin(lam*dl), -1j*self.H[0,1]*np.sin(lam*dl)],[-1j*np.conjugate(self.H[0,1])*np.sin(lam*dl), lam*np.cos(lam*dl) + 1j*self.H[0,0]*np.sin(lam*dl)]], dtype = np.complex_)
        return dH
