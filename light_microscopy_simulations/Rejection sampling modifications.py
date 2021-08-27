import numpy as np


def sample_random_curve(w_t):
    considered_times = np.linspace(1,config.T,6)

    # times
    positions = rejection_sampling(0, w_t)
    for t in considered_times[1:]:
        positions = np.append(positions, rejection_sampling(t, w_t), 0)
    
    n = len(considered_times) - 1
    r = config.interpolation_pooling_number

    potential_nodes = np.zeros([n,r,2])
    interpolation_times = np.empty
    
    for ii in range(n):
        t = np.round((considered_times[ii+1] + considered_times[ii])/2)
        np.append(interpolation_times, t)
        x_0 = (positions[ii+1,:] + positions[ii])/2
        M = max_conv_density(t, w_t, x_0)
        for jj in range(r):
            potential_nodes[ii,jj,:] = modified_rejection_sampling(t, w_t, x_0, M)
    
    considered_times = np.sort(np.append(considered_times, interpolation_times))

    pos = np.zeros(positions.size + n)
    pos[0::2,:] = positions

    #There should be r^n curves
    for jj in range(r**n):
        for ll in range(n):
            pos[2*ll+1,:] = potential_nodes[ll,np.remainder(np.floor(jj/r^ll),r),:]    
        new_curve =  classes.curve(considered_times/(config.T-1), pos)
        np.append(intermediate_pooling_curves, new_curve)
        np.append(intermediate_energies, F(w_t,new_curve))

    idx = np.argmin(intermediate_energies)
    rand_curve = intermediate_pooling_curves[idx]

    return rand_curve, len(considered_times)

def F(w_t, curve):
    # Define the energy here to evaluate the crossover children
    return -curve.integrate_against(w_t)/curve.energy()

def max_conv_density(t, w_t, x_0, resolution = 0.01):
    k = config.k 
    x = np.linspace(0, 1, round(1/resolution))
    y = np.linspace(0, 1, round(1/resolution))
    X, Y = np.meshgrid(x, y)
    XY = np.array([np.array([xx, yy]) for yy, xx in it.product(y, x)])
    evaluations = w_t._density_transformation(t, w_t.eval(t, XY))*np.exp(-( np.square( XY[0,:] - x_0[0] ) + np.square( XY[1,:] - x_0[1] ) )/(2*k*k))
    return np.max(evaluations)

def modified_rejection_sampling(t, w_t, x_0, M):
    """ Rejection sampling over a density defined by the dual variable.

    Parameters
    ----------
    t : int
        Index of time sample. Takes values between 0,1,...,T. Where (T+1) is
        the total number of time samples of the inverse problem.
    w_t : :py:class:`src.classes.dual_variable`
        Dual variable associated with the current iterate.

    Returns
    -------
    numpy.ndarray
        A random point in Ω = [0,1]^2.
    """
    iter_reasonable_threshold = 1000
    iter_index = 0
    while iter_index < iter_reasonable_threshold:
        # sample from uniform distribution on the support of w_t as a density.
        reasonable_threshold = 1000
        i = 0
        while i < reasonable_threshold:
            x = np.random.rand()
            y = np.random.rand()
            sample = np.array([[x, y]])
            h = w_t._density_transformation(t, w_t.eval(t, sample))
            if h > 0:
                break
            else:
                i = i + 1
        if i == reasonable_threshold:
            sys.exit('It is not able to sample inside the support of w_t')
        # sample rejection sampling
        u = np.random.rand()
        #We normalise by e^(1+\epsilon) - 1 since this is the maximum value attained by the density transformation. 
        if u < h*np.exp(-( np.square(x - x_0[0]) + np.square(y - x_0[1]) )/(2*config.k*config.k)):
            # accept
            return sample
        else:
            # reject
            iter_index = iter_index+1
    sys.exit(('The rejection_sampling algorithm failed to find sample in {} ' +
             'iterations').format(iter_index))