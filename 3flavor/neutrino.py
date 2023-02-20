import numpy as np
import matplotlib.pyplot as plt

class Neutrino:
    # Initialization
    #
    # Parameters:
    # E: neutrino energy
    # theta_0: initial angle
    # kind: 'e' for electron neutrino, 'x' and 'y' for linear combinations of muon and tau neutrinos
    # 
    # Returns:
    # Neutrino object with the following attributes:
    # E: neutrino energy
    # theta_0: initial angle
    # psi: initial state vector that is either [1,0] for electron neutrino or [0,1] 'x' neutrino
    def __init__(self, E, theta_0, kind):
        self.E = E
        self.theta_0 = theta_0
        if kind == 'e':
            self.psi = np.array([1,0,0])
        elif kind == 'x':
            self.psi = np.array([0,1,0])
        else:
            self.psi = np.array([0,0,1])
    
    # String representation of the object
    #
    # Parameters:
    # None 
    #   
    # Returns:
    # String representation of the object
    def __str__(self):
        return 'Neutrino object with energy {} and initial angle {}'.format(self.E, self.theta_0)
    
    # Geometry factor
    #
    # Parameters:
    # R_nu: radius of the neutrino sphere
    # r: current radius
    #
    # Returns:
    # Geometrical factor
    def geometrical_factor(self, R_nu, r):
        return np.cos(np.arcsin(R_nu/r*np.sin(self.theta_0)))
    
    # State evolution
    #
    # Parameters:
    # dH: Hamiltonian change
    #   
    # Returns: 
    # Updated state vector
    def evolve(self, dH):
        psi = np.dot(dH, self.psi)
        self.psi = psi


