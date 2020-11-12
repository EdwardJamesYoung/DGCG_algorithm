import numpy as np
import DGCG

# Simulation parameters
T = 30
time_samples = np.linspace(0,1,T)
K = np.ones(T, dtype=int)*18

# Kernels to define the forward operators

def cut_off(s):
    # A one dimentional cut-off function that is twice differentiable, monoto-
    # nous and fast to compute.
    # Input: s in Nx1 numpy array, representing 1-D evaluations.
    # Output: a Nx1 numpy array evaluating the 1-D cutoff along the input s.
    # the cut-off threshold is the width of the transition itnerval from 0 to 1.
    cutoff_threshold = 0.1
    transition = lambda s: 10*s**3 - 15*s**4 + 6*s**5
    val = np.zeros(s.shape)
    for i in range(len(s)):
        if s[i]>= cutoff_threshold and s[i]<= 1-cutoff_threshold:
            val[i]=1
        elif s[i]<cutoff_threshold and s[i]>= 0:
            val[i]= transition(s[i]/cutoff_threshold)
        elif s[i]>1-cutoff_threshold and s[i]<=1:
            val[i]= transition(((1-s[i]))/cutoff_threshold)
        else:
            val[i] = 0
    return val

def D_cut_off(s):
    # The derivative of the defined cut_off function.
    # Same input and output sizes.
    cutoff_threshold = 0.1
    D_transition = lambda s: 30*s**2 - 60*s**3 + 30*s**4
    val = np.zeros(s.shape)
    for i in range(len(s)):
        if s[i]< cutoff_threshold and s[i]>=0:
            val[i]= D_transition(s[i]/cutoff_threshold)/cutoff_threshold
        elif s[i]<=1 and s[i] >= 1-cutoff_threshold:
            val[i]= -D_transition(((1-s[i]))/cutoff_threshold)/cutoff_threshold
        else:
            val[i]=0
    return val

# Implementation of the test functions

def Archimedian_spiral(t,a,b):
    return np.array([(a+b*t)*np.cos(t), (a+b*t)*np.sin(t)])

available_samples = np.array([Archimedian_spiral(t,0,0.2) for t in
                                            np.arange(K[0])]) # K[i] are equal.
sampling_method = [available_samples for t in range(T)]

def test_func(t,x): # φ_t(x)
    # Input: t∈[0,1,2,...,T-1]
    #        x numpy array of size Nx2, representing a list of spatial points
    #            in R^2.
    # Output: NxK numpy array, corresponding to the  test function evaluated in
    #         the set of spatial points.

    # # complex exponential test functions
    expo = lambda s: np.exp(-2*np.pi*1j*s)
    # # The evaluation points for the expo functions, size NxK.
    evals = x@sampling_method[t].T
    # # The considered cutoff, as a tensor of 1d cutoffs (output: Nx1 vector)
    h = 0.1
    cutoff = cut_off(x[:,0:1])*cut_off(x[:,1:2])
    # return a np.array of vectors in H_t, i.e. NxK numpy array.
    return expo(evals)*cutoff

def grad_test_func(t,x): # ∇φ_t(x)
    # Gradient of the test functions before defined. Same inputs.
    # Output: 2xNxK numpy array, where the first two variables correspond to
    #         the dx part and dy part respectively.
    # #  Test function to consider
    expo = lambda s: np.exp(-2*np.pi*1j*s)
    # # The sampling locations defining H_t
    S = sampling_method[t]
    # # Cutoffs
    h = 0.1
    cutoff_1 = cut_off(x[:,0:1])
    cutoff_2 = cut_off(x[:,1:2])
    D_cutoff_1 = D_cut_off(x[:,0:1])
    D_cutoff_2 = D_cut_off(x[:,1:2])
    # # preallocating
    N = x.shape[0]
    output = np.zeros((2,N,K[t]), dtype = 'complex')
    # # Derivative along each direction
    output[0] = expo(x@S.T)*cutoff_2*(
                                -2*np.pi*1j*cutoff_1@S[:,0:1].T + D_cutoff_1)
    output[1] = expo(x@S.T)*cutoff_1*(
                                -2*np.pi*1j*cutoff_2@S[:,1:2].T + D_cutoff_2)
    return output


DGCG.set_parameters(time_samples, K, test_func, grad_test_func)

# Generate data. Simple two crossing curves

## First curve, straight one.
initial_position_1 = [0.2, 0.2]
final_position_1   = [0.8, 0.8]
positions_1 = np.array( [initial_position_1, final_position_1] )
times_1 = np.array([0,1])
curve_1 = DGCG.curves.curve(times_1, positions_1)

## Second curve, with a kink
initial_position_2 = [0.8, 0.2]
middle_position_2  = [0.5, 0.5]
final_position_2   = [0.6, 0.8]
times_2 = np.array([0, 0.5, 1])
positions_2 = np.array([initial_position_2, middle_position_2, final_position_2])
curve_2 = DGCG.curves.curve(times_2, positions_2)

## Include these curves inside a measure, with respective intensities
intensity_1 = 1
intensity_2 = 1.5
measure = DGCG.curves.measure()
measure.add(curve_1, intensity_1)
measure.add(curve_2, intensity_2)
### uncomment to see the animated curve
#measure.animate()

# Simulate the measurements generated by this curve
data = DGCG.operators.K_t_star_full(measure)
## uncomment to see the backprojected data
DGCG.config.f_t = data
dual_variable = DGCG.operators.w_t(DGCG.curves.measure())
ani_1 = dual_variable.animate(measure = measure, block = True)

# Add noise to the measurements
noise_level = 0.2 # 20% of noise
noise_vector = np.random.randn(*np.shape(data))
data_H_norm = np.sqrt(DGCG.operators.int_time_H_t_product(data,data))
data_noise = data + noise_vector*noise_level/data_H_norm

## uncomment to see the backprojected data
DGCG.config.f_t = data_noise
dual_variable = DGCG.operators.w_t(DGCG.curves.measure())
ani_2 = dual_variable.animate(measure = measure, block = True)

# Use the DGCG solver
alpha = 0.2
beta = 0.2

current_measure = DGCG.solve(data_noise, alpha, beta)
