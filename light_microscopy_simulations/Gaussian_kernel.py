import numpy as np

def create_grid(n):
    xx, yy = np.meshgrid(np.linspace(0,1,n),np.linspace(0,1,n))
    return np.stack((xx.flatten(), yy.flatten()))

class GaussianKernel:
    def __init__(self, Resolution, sigma):
        self.Resolution = Resolution
        self.sigma = sigma
        self.grid = create_grid(Resolution)
    
    def gauss2d(self,x,m):
        """
        Parameters:
        m: numpy.ndarray of shape (2,1) 
        x: numpy.ndarray of shape (2,n) for n arbitrary
        
        Returns: 
        numpy.ndarray of shape (n,) - j^th entry is the value of the 2D Gaussian pdf with mean m 
            and s.d. sigma evaluated at the point x[:,j]
        """
        return np.exp( -( np.square( x[0,:] - m[0] ) + np.square( x[1,:] - m[1] ) )/(2*self.sigma*self.sigma) )/(2*np.pi*self.sigma*self.sigma)
        
        
    def eval(self, t, x):
        """
        Parameters:
        t: int - the time sample of evaluation
        x: numpy.ndarray of size (N,2) - the spacial points at which we 
            evaluate the function

        NOTE: our forward operator is independant of time. 
        
        Returns:
        numpy.ndarray of size (N, Res^2) - the Gaussian function shifted to being
            centered at each of the grid points 
        """
        return 10*np.stack([self.gauss2d(self.grid,x[k,:]) for k in range(x.shape[0])])/self.Resolution


    def grad(self,t,x): 
        """
        Parameters:
        t: int - the time sample of evaluation 
            Note that again this is not used anywhere in the function
            
        x: numpy.ndarray of size (N,2)  - a list of spatial points at which 
            the gradient is to be calculated
            
        returns:
        numpy.ndarray of size (2,N,Res^2) - a list of the spacial derivative of 
            component of the function in both the x and y directions, at each of the
            inputed spacial points
        """
        x_disp = np.stack([ self.grid[0,:] - x[k,0] for k in range(x.shape[0]) ] )
        y_disp = np.stack([ self.grid[1,:] - x[k,1] for k in range(x.shape[0]) ] )
    
        Gaussian_scaling = self.eval(t,x)/(self.sigma*self.sigma)
        return np.array([x_disp*Gaussian_scaling, y_disp*Gaussian_scaling])