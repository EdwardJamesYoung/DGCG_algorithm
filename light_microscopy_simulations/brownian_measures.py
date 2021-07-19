import numpy as np

import sys
import os

sys.path.insert(0, os.path.abspath('..'))
from src import DGCG

def generate_brownian_curve(T,S,centre = np.array([0.5,0.5])):    
    positions = np.ndarray(shape = (T,2))
    multiplier = S/np.sqrt(T-1)
    positions[0] = centre
    
    for t in range(1,T):
        positions[t] = positions[t-1] + multiplier*np.random.randn(1,2)
    
    positions = np.clip(positions,0,1)
    return positions

def generate_brownian_measure(T,scalings,centres,intensities):
    """
    Inputs:
    T - int - number of time steps over which the motion is to be simulated
    scalings - np.ndarray of shape (n,1) - the scale of the each of the motions
    centres - np.ndarray of shape (n,2) - the startin point of each of the motions
    intensities - np.ndarray of shape (2,1) - the intensities of each of the motions
    """
    brownian_measure = DGCG.classes.measure()
    n = scalings.shape[0]
    for ii in range(n):
        brownian_curve = DGCG.classes.curve(generate_brownian_curve(T,scalings[ii],centre=centres[ii,:]))
        brownian_measure.add(brownian_curve,intensities[ii]*brownian_curve.energy())
    return brownian_measure
