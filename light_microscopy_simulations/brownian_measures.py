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
    return positions

def generate_brownian_measure(T,scalings,centres,intensities):
    """
    Parameters:
    T - int - number of time steps over which the motion is to be simulated
    scalings - np.ndarray of shape (n,1) - the scale of the each of the motions
    centres - np.ndarray of shape (n,2) - the starting point of each of the motions
    intensities - np.ndarray of shape (2,1) - the intensities of each of the motions

    Returns:
    A classes.measure object created by adding together n Brownian motions, with the centres,
        scalings, and intensities as given by the parameters.
    """
    brownian_measure = DGCG.classes.measure()
    n = scalings.shape[0]
    for ii in range(n):
        brownian_curve = DGCG.classes.curve(np.clip(generate_brownian_curve(T,scalings[ii],centre=centres[ii,:]),0,1))
        brownian_measure.add(brownian_curve,intensities[ii]*brownian_curve.energy())
    return brownian_measure

def generate_brownian_bridge_curve(T,S):
    positions = np.ndarray(shape = (T,2))
    multiplier = S/np.sqrt(T-1)
    positions[0] = np.array([0,0])
    
    for t in range(1,T):
        positions[t] = positions[t-1] + multiplier*np.random.randn(1,2)
    
    positions = positions - positions[T-1, np.newaxis] * np.stack([np.linspace(0,1,T),np.linspace(0,1,T)]).T
    return positions

def generate_directed_brownian_measure(T,scalings,start_points,end_points,intensities):
    """
    Parameters:
    T - int - number of time steps over which the motion is to be simulated
    scalings - np.ndarray of shape (n,1) - the scale of the each of the motions
    start_points - np.ndarray of shape (n,2) - the starting point of each of the motions
    end_points - np.ndarray of shape (n,2) - the ending points of each of the motions
    intensities - np.ndarray of shape (2,1) - the intensities of each of the motions

    Returns:
    A classes.measure object created by adding together n directed Brownian motions, with
        the starting points, ending points, scalings, and intensities as given by the parameters.
        
        Here, a directed Brownian motion is the sum of a Brownian bridge with a linear motion 
        from a start point to an end point.
    """
    
    directed_measure = DGCG.classes.measure()
    n = scalings.shape[0]
    for ii in range(n):
        positions = np.linspace(start_points[ii,:],end_points[ii,:],T) + generate_brownian_bridge_curve(T,scalings[ii])
        positions = np.clip(positions,0,1)
        curve = DGCG.classes.curve(positions)
        directed_measure.add(curve,intensities[ii]*curve.energy())
    return directed_measure

def generate_constrained_brownian_curve(T,S,centre = np.array([0.5,0.5])):
    positions = np.ndarray(shape = (T,2))
    multiplier = S/np.sqrt(T-1)
    positions[0] = centre
    
    for t in range(1,T):
        positions[t] = positions[t-1] + multiplier*np.random.randn(1,2)
        overspill = positions[t] - np.clip(positions[t],0,1)
        positions[t] = positions[t] -2*overspill
    return positions

def generate_reflective_brownian_measure(T,scalings,centres,intensities):
    """
    Parameters:
    T - int - number of time steps over which the motion is to be simulated
    scalings - np.ndarray of shape (n,1) - the scale of the each of the motions
    centres - np.ndarray of shape (n,2) - the starting point of each of the motions
    intensities - np.ndarray of shape (2,1) - the intensities of each of the motions

    Returns:
    A classes.measure object created by adding together n Brownian motions, with the centres,
        scalings, and intensities as given by the parameters. We impose reflective boundary 
        conditions on the motions, by reflecting a particle of the edges of the box [0,1]^2
    """
    reflective_brownian_measure = DGCG.classes.measure()
    n = scalings.shape[0]
    for ii in range(n):
        brownian_curve = DGCG.classes.curve(generate_constrained_brownian_curve(T,scalings[ii],centre=centres[ii,:]),0,1)
        reflective_brownian_measure.add(brownian_curve,intensities[ii]*brownian_curve.energy())
    return reflective_brownian_measure

def L2norm_noise_model(data,noise_level):
    """
    Parameters: 
    noise_level - float - The ratio of the integrated squared norm of the noise vector to the integrated squared norm of the signal vector
    measure - The measure from which the data is to be generated.

    Returns:
    numpy.ndarray of shape equal to that of DGCG.operators.K_t_star_full(measure) - The noise to be added to the data
    """    
    #Generate random noise vector using a Gaussian distribution.
    noise = np.random.randn(*data.shape)
    #Normalise the noise vector.
    noise = noise/np.sqrt(DGCG.operators.int_time_H_t_product(noise,noise))
    #Scale the noise vector back up.
    noise = noise*noise_level*np.sqrt(DGCG.operators.int_time_H_t_product(data,data))

    return noise

def PSNR_noise_model(data,PSNR):
    """
    Parameters:
    PSNR - float - The PSNR value of the noise
    measure - the measure from which the data is to be generated

    Returns:
    numpy.ndarray of shape equal to that of the data 

    """
    signal_strength = np.median(np.max(data, axis = 1) - np.min(data, axis = 1))
    s = signal_strength* 10**(-PSNR/20)
    noise = s*np.random.randn(*data.shape)
    return noise