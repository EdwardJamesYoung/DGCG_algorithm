"""Container of the used classes of the module.
"""
# Standard imports
import copy
import numpy as np
# Plotting imports
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Local imports
from . import misc, config, checker
from . import operators as op

# Module methods


class curve:
    """Piecewise linear continuous curves in the domain Ω.

    To There are two ways to initialize a curve. Either input a single
    numpy.ndarray of size (T,2), representing a set of ``T`` spatial points,
    the produced curve will take N uniformly taken time samples.

    Alternative, initialize with two arguments, the first one a one dimentional
    ordered list of time samples of size T, and a set of corresponding
    numpy.ndarray of size (T,2).

    Attributes
    ----------
    spatial_points : numpy.ndarray
        (T,2) sized array with ``T`` the number of time samples. Corresponds to
        the position of the curve at each time sample.
    time_samples : numpy.ndarray
        (T,) sized array corresponding to each time sample.
    """
    def __init__(self, *args):
        assert len(args) <= 2
        if len(args) == 1:
            # case in which just positions were used
            space = args[0]
            assert checker.is_in_space_domain(space)
            time = np.linspace(0, 1, len(space))
            self.time_samples = time
            self.spatial_points = space
            if len(space) != len(config.time):
                self.set_times(config.time)
        else:
            # case in which time and positions were used
            space = args[1]
            time = args[0]
            assert checker.is_in_space_domain(space) \
                   and all(time >= 0) and all(time <= 1)
            self.time_samples = time
            self.spatial_points = space
            self.set_times(config.time)

    def __add__(self, curve2):
        new_curve = curve(self.time_samples,
                          self.spatial_points + curve2.spatial_points)
        return new_curve

    def __sub__(self, curve2):
        new_curve = curve(self.time_samples,
                          self.spatial_points - curve2.spatial_points)
        return new_curve

    def __mul__(self, factor):
        new_curve = curve(self.time_samples, self.spatial_points*factor)
        return new_curve

    def __rmul__(self, factor):
        new_curve = curve(self.time_samples, self.spatial_points*factor)
        return new_curve

    def draw(self, tf=1, ax=None, color=[0.0, 0.5, 1.0], plot=True):
        """Method to draw the curve.

        Using `matplotlib.collections.LineCollection`, this method draws the
        curve as a collection of segments, whose transparency indicates the
        time of the drawn curve. It also returns the segments and their
        respective colors.

        Parameters
        ----------
        tf : float, optional
            value in (0,1] indicating until which time the curve will be drawn.
            Default 1.
        ax : matplotlib.axes.Axes, optional
            An axes object to which to include the drawing of the curve.
            Default None
        color : list[float], optional
            Length-3 list of the RGB color to give to the curve. Default
            [0.0, 0.5, 1.0]
        plot : bool, optional
            Switch to draw or not the curve.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the drawn curve
        segments_colors : (numpy.ndarray, numpy.ndarray)
            A tuple with the segments describing the curve on the first entry,
            and the RGBA colors of them in the second entry
        """
        # First we supersample the whole curve such that there are not jumps
        # higher than a particular value
        supersampl_t, supersampl_x = misc.supersample(self, max_jump=0.01)
        # Drop all time samples after tf
        value_at_tf = self.eval(tf)
        index_tf = np.argmax(supersampl_t >= tf)
        supersampl_x = supersampl_x[:index_tf]
        supersampl_t = supersampl_t[:index_tf]
        supersampl_t = np.append(supersampl_t, tf)
        supersampl_x.append(value_at_tf.reshape(-1))
        # Reduce the set of points and times to segments and times, restricted
        # to the periodic domain.
        segments = [[supersampl_x[j], supersampl_x[j+1]]
                    for j in range(len(supersampl_x)-1)]
        # Use the LineCollection class to print using segments and to assign
        # transparency or colors to each segment
        line_segments = LineCollection(segments)
        # set color
        lowest_alpha = 0.2
        color = np.array(color)
        rgb_color = np.ones((len(segments), 4))
        rgb_color[:, 0:3] = color
        rgb_color[:, 3] = np.linspace(lowest_alpha, 1, len(segments))
        line_segments.set_color(rgb_color)
        start_color = np.zeros((1, 4))
        start_color[:, 0:3] = color
        if len(segments) <= 1:
            start_color[:, 3] = 1
        else:
            start_color[:, 3] = lowest_alpha
        # plotting
        if plot:
            ax = ax or plt.gca()
            ax.add_collection(line_segments)
            ax.scatter(self.spatial_points[0, 0], self.spatial_points[0, 1],
                       c=start_color, marker='x', s=0.4)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return ax, (segments, rgb_color)
        return None, (segments, rgb_color)

    def eval(self, t):
        """Evaluate the curve at a certain time.

        Parameters
        ----------
        t : list[float] or float
            time values in ``[0,1]``.

        Returns
        -------
        positions : numpy.ndarray
            (N,2) sized array representing ``N`` different points in ``R^2``.
            ``N`` corresponds to the number of input times.
        """
        assert t <= 1 and t >= 0
        if isinstance(t, (float, int)):
            t = np.array(t).reshape(1)
        N = len(t)
        evals = np.zeros((N, 2))
        for i, tt in enumerate(t):
            if tt == 0:
                evals[i, :] = self.spatial_points[0, :]
            else:
                index = np.argmax(np.array(self.time_samples) >= t[i])
                ti = self.time_samples[index-1]
                tf = self.time_samples[index]
                xi = self.spatial_points[index-1, :]
                xf = self.spatial_points[index, :]
                evals[i, :] = (tt-ti)/(tf-ti)*xf + (tf-tt)/(tf-ti)*xi
        return evals

    def eval_discrete(self, t):
        """ Evaluate the curve at a certain time node.

        Parameters
        ----------
        t : int
            The selected time sample, in 0,1,...,T-1.

        Returns
        -------
        numpy.ndarray
            A single spatial point represented by a (1,2) array.
        """
        assert checker.is_valid_time(t)
        return self.spatial_points[t:t+1]

    def integrate_against(self, w_t):
        """Method to integrate a dual variable along this curve.

        Parameters
        ----------
        w_t : :py:class:`src.operators.w_t`
            The dual variable to integrate against

        Returns
        -------
        float
            The integral of w_t along the curve.
        """
        assert isinstance(w_t, op.w_t)
        output = 0
        weights = config.time_weights
        for t in range(config.T):
            output += weights[t]*w_t.eval(t, self.eval_discrete(t))
        return output.reshape(1)[0]

    def H1_seminorm(self):
        """Computes the ``H^1`` seminorm of the curve

        Returns
        -------
        float
        """
        diff_t = np.diff(config.time)
        diff_points = np.diff(self.spatial_points, axis=0)
        squares_divided_times = (1/diff_t)@diff_points**2
        return np.sqrt(np.sum(squares_divided_times))

    def L2_norm(self):
        """Computes the ``L^2`` norm of the curve

        Returns
        -------
        float
        """
        diff_times = self.time_samples[1:]-self.time_samples[:-1]
        x = self.spatial_points
        x_vals = x[:-1, 0]**2 + x[:-1, 0]*x[1:, 0] + x[1:, 0]**2
        L2_norm_x = np.sum(diff_times/3*x_vals)
        y_vals = x[:-1, 1]**2 + x[:-1, 1]*x[1:, 1] + x[1:, 1]**2
        L2_norm_y = np.sum(diff_times/3*y_vals)
        return np.sqrt(L2_norm_x + L2_norm_y)

    def H1_norm(self):
        """Computes the ``H^1`` norm of this curve.

        Returns
        -------
        float
        """
        return self.L2_norm() + self.H1_seminorm()

    def energy(self):
        """Computes the Benamou-Brenier with Total variation energy.

        Returns
        -------
        float
        """
        return config.beta/2*self.H1_seminorm()**2 + config.alpha

    def set_times(self, new_times):
        """Method to change the ``time_samples`` member,

        It changes the vector of time samples by adjusting accordingly the
        ``spatial_points`` member,

        Parameters
        ----------
        new_times : numpy.ndarray
            1 dimensional array with new times to have the curvee defined in.

        Returns
        -------
        None

        """
        assert isinstance(new_times, np.ndarray) and \
               np.min(new_times) >= 0 and np.max(new_times) <= 1
        # Changes the time vector to a new one, just evaluates and set the new 
        # nodes, nothing too fancy.
        new_locations = np.array([self.eval(t).reshape(-1) for t in new_times])
        self.time_samples = new_times
        self.spatial_points = new_locations
        self.quads = None

class curve_product:
    """Elements of a weighted product space of curve type objects.

    It can be initialized with empty arguments, or via the keyworded arguments
    `curve_list` and `weights`.

    Attributes
    ----------
    weights : list[float]
        Positive weights associated to each space.
    curves : list[:py:class:`src.classes.curve`]
        List of curves
    """
    def __init__(self, curve_list=None, weights=None):
        if curve_list is None or weights is None:
            self.weights = []
            self.curve_list = []
        else:
            if len(curve_list) == len(weights) and all(w > 0 for w in weights):
                self.weights = weights
                self.curve_list = curve_list
            else:
                raise Exception("Wrong number of weights and curves in the " + 
                                "curve list, or weights non-positive")

    def __add__(self, curve_list2):
        # Warning, implemented for curve_lists with the same weights
        new_curve_list = [curve1+curve2 for curve1, curve2 in
                          zip(self.curve_list, curve_list2.curve_list)]
        return curve_product(new_curve_list, self.weights)

    def __sub__(self, curve_list2):
        new_curve_list = [curve1-curve2 for curve1, curve2 in
                          zip(self.curve_list, curve_list2.curve_list)]
        return curve_product(new_curve_list, self.weights)

    def __mul__(self, factor):
        new_curve_list = [curve1*factor for curve1 in self.curve_list]
        return curve_product(new_curve_list, self.weights)

    def __rmul__(self, factor):
        new_curve_list = [curve1*factor for curve1 in self.curve_list]
        return curve_product(new_curve_list, self.weights)

    def H1_norm(self):
        """Computes the weighted product :math:`H^1` norm.

        Returns
        -------
        float
        """
        output = 0
        for weight, curve in zip(self.weights, self.curve_list):
            output += weight*curve.H1_norm()/len(self.curve_list)
        return output

    def to_measure(self):
        """Cast this objet into :py:class:`src.classes.measure` """
        new_measure = measure()
        for weight, curve in zip(self.weights, self.curve_list):
            new_measure.add(curve, weight)
        return new_measure


class measure:
    """Sparse measures composed of a finite weighted sum of Atoms.

    Initializes with empty arguments to create the zero measure.

    Attributes
    ----------
    curves : list[:py:class:`src.classes.curve`]
        List of member curves.
    weights : numpy.ndarray
        Array of positive weights associated to each curve.
    energies : numpy.ndarray
        Array of stored Benamou-Brenier energies associated to each curve.
        See :py:meth:`src.classes.curve.energy`.
    main_energy : float
        The Tikhonov energy of the measure.

    Notes
    -----
    As described in the theory, an Atom is a Dirac delta on a curve with a
    respective weight. This weight is defined by 1/energy of the curve.
    """
    def __init__(self):
        self.curves = []
        self.energies = np.array([])
        self.weights = np.array([])
        self.main_energy = None

    def add(self, new_curve, new_weight):
        """Include a new curve with associated weight into the measure.

        Parameters
        ----------
        new_curve : :py:class:`src.classes.curve`
            Curve to be added.
        new_weight : float
            Positive weight to be added.

        Returns
        -------
        None
        """
        if new_weight > config.measure_coefficient_too_low:
            self.curves.extend([new_curve])
            self.energies = np.append(self.energies,
                                      new_curve.energy())
            self.weights = np.append(self.weights, new_weight)
            self.main_energy = None

    def __add__(self, measure2):
        new_measure = copy.deepcopy(self)
        new_measure.curves.extend(copy.deepcopy(measure2.curves))
        new_measure.energies = np.append(new_measure.energies,
                                         measure2.energies)
        new_measure.weights = np.append(new_measure.weights, measure2.weights)
        new_measure.main_energy = None
        return new_measure

    def __mul__(self, factor):
        if factor <= 0:
            raise Exception('Cannot use a negative factor for a measure')
        new_measure = copy.deepcopy(self)
        new_measure.main_energy = None
        for i in range(len(self.weights)):
            new_measure.weights[i] = new_measure.weights[i]*factor
        return new_measure

    def __rmul__(self, factor):
        return self*factor

    def modify_weight(self, curve_index, new_weight):
        """Modifies the weight of a particular Atom/curve

        Parameters
        ----------
        curve_index : int
            Index of the target curve stored in the measure.
        new_weight : float
            Positive new weight.

        Returns
        -------
        None
        """
        self.main_energy = None
        if curve_index >= len(self.curves):
            raise Exception('Trying to modify an unexistant curve! The given'
                            + 'curve index is too high for the current array')
        if new_weight < config.measure_coefficient_too_low:
            del self.curves[curve_index]
            self.weights = np.delete(self.weights, curve_index)
            self.energies = np.delete(self.energies, curve_index)
        else:
            self.weights[curve_index] = new_weight

    def integrate_against(self, w_t):
        """Integrates the measure against a dual variable.

        Parameters
        ----------
        w_t : :py:class:`src.operators.w_t`

        Returns
        -------
        float
        """
        assert isinstance(w_t, op.w_t)
        # Method to integrate against this measure. 
        integral = 0
        for i, curv in enumerate(self.curves):
            integral += self.weights[i]/self.energies[i] * \
                    curv.integrate_against(w_t)
        return integral

    def spatial_integrate(self, t, target):
        """Spatially integrates the measure against a function for fixed time.

        Parameters
        ----------
        t : int
            Index of time sample in ``0,1,...,T-1``.
        target : callable[numpy.ndarray, float]
            A function that takes values on the 2-dimensional domain and
            returns a real number.

        Returns
        -------
        float
        """
        val = 0
        for i in range(len(self.weights)):
            val = val + self.weights[i]/self.energies[i] * \
                    target(self.curves[i].eval_discrete(t))
        return val
 
    def to_curve_product(self):
        """Casts the measure into a :py:class:`src.classes.curve_product`.

        Returns
        -------
        None
        """
        return curve_product(self.curves, self.weights)

    def get_main_energy(self):
        """Computes the Tikhonov energy of the Measure.

        It also stores it as a member of the measure.

        Returns
        -------
        float
        """
        if self.main_energy is None:
            self.main_energy = op.main_energy(self, config.f_t)
            return self.main_energy
        else:
            return self.main_energy

    def draw(self, ax=None):
        """Draws the measure.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            axes to include the drawing. Defaults to None.

        Returns
        -------
        matplotlib.axes.Axes
            The modified, or new, axis with the drawing.
        """
        num_plots = len(self.weights)
        intensities = self.weights/self.energies
        'get the brg colormap for the intensities of curves'
        colors = plt.cm.brg(intensities/max(intensities))
        ax = ax or plt.gca()
        for i in range(num_plots):
            self.curves[i].draw(ax=ax, color=colors[i, :3])
        plt.gca().set_aspect('equal', adjustable='box')
        'setting colorbar'
        norm = mpl.colors.Normalize(vmin=0, vmax=max(intensities))
        cmap = plt.get_cmap('brg', 100)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ticks=np.linspace(0, max(intensities), 10))
        return ax

    def animate(self, filename=None, show=True, block=False):
        """Method to create an animation representing the measure object.

        Uses ``matplotlib.animation.FuncAnimation`` to create a video
        representing the measure object, where each curve, and its respective
        intensity is represented. The curves are ploted on time, and the color
        of the curve represents the respective intensity.  It is possible to
        output the animation to a ``.mp4`` file if ``ffmpeg`` is available.

        Parameters
        ----------
        filename : str, optional
            A string to save the animation as ``.mp4`` file. Default None
            (no video is saved).
        show : bool, optional
            Switch to indicate if the animation should be immediately shown.
            Default True.
        frames : int, optional
            Number of frames considered in the animation. Default 51.

        Returns
        -------
        None
        """
        animation = misc.Animate(self, frames=51, filename=filename, show=show)
        animation.draw()

    def reorder(self):
        """Reorders the curves and weights of the measure.

        Reorders the elements such that they have increasing intensity.
        The intensity is defined as ``intensity = weight/energy``

        Returns
        -------
        None
        """
        intensities = self.weights/self.energies
        new_order = np.argsort(intensities)
        new_measure = measure()
        for idx in new_order:
            new_measure.add(self.curves[idx], self.weights[idx])
        self.curves = new_measure.curves
        self.weights = new_measure.weights
        self.energies = new_measure.energies


if __name__ == '__main__':
    pass
