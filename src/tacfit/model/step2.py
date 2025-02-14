import numpy as np
import numpy.typing as npt
import scipy


def model_step2(t: npt.NDArray[np.float64],
                in_func: npt.NDArray[np.float64],
                **kwargs: float) -> npt.NDArray[np.float64]:
    """Solves the model where the input response function is assumed to be a
    2-step function.
    This function calculates the convolution of a sampled input function with
    a 2-step function. The step function has value amp1+amp2 on the interval
    [0, extent1), value amp2 on the interval [extent1, extent2) and value 0 on
    the interval [extent2, infinity).
    The convolution is evaluated at the same time points as the sampled
    input function and returned as a list.
    The convolution is performed numerically using scipy.integrate.quad and
    the input function is interpolated linearly between sample points.

    Arguments:
    t       --  The time points of the input function samples.
    in_func --  The input function samples.
    amp1    --  The amplitude of the step function on [0, extent1).
    extent1 --  The length of the first step function.
    amp2    --  The amplitude of the step function on [0, extent2).
    extent2 --  The length of the second step function.

    Return value:
    A list containing the modeled values at each time point.
    """

    res = np.zeros_like(t)

    # Unpack parameters
    ext1 = kwargs['extent1']
    ext2 = kwargs['extent2']
    amp1 = kwargs['amp1']
    amp2 = kwargs['amp2']

    # The function to be integrated is just the amplitudes times the integral
    # of the input function. The amplitudes will be multiplied after
    # integration.
    def integrand(tau: float):
        return np.interp(tau, t, in_func)

    for i in range(0, res.size):
        # For each time point the integrand is integrated.

        # We split the integral up into the two step-functions to avoid
        # discontinuity problems

        # First integral from max(0, t-ext2) to max(0, t-ext1)
        y1 = scipy.integrate.quad(integrand,
                                  max(0.0, t[i] - ext2),
                                  max(0.0, t[i] - ext1))

        # Second integral from max(0, t-ext1) to t
        y2 = scipy.integrate.quad(integrand,
                                  max(0.0, t[i] - ext1),
                                  t[i])

        # Multiply by the amplitudes
        res[i] = amp2*y1[0] + (amp1 + amp2)*y2[0]

    return res
