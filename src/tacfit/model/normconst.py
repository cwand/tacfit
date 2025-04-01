import math

import numpy as np
import numpy.typing as npt
import scipy


def irf_normconst(
        t: npt.NDArray[np.float64],
        **kwargs: float) -> npt.NDArray[np.float64]:

    amp1 = kwargs['amp1']
    amp2 = kwargs['amp2']
    ext1 = kwargs['extent1']
    wid1 = kwargs['width1']

    # Calculate normcdf using error function (seems to be much quicker)
    tt = (t - ext1) / (math.sqrt(2.0) * wid1)
    cdf = 0.5 * (1.0 + scipy.special.erf(tt))

    # Calculate input response function
    return np.array(amp2 + (amp1 - amp2) * (1.0 - cdf))


def model_normconst(
        t_in: npt.NDArray[np.float64],
        in_func: npt.NDArray[np.float64],
        t_out: npt.NDArray[np.float64],
        **kwargs: float) -> npt.NDArray[np.float64]:
    """Solves the model where the impulse response function starts at the value
    amp1, transitions smoothly to a new value amp2 around the time point
    extent1, and then stays constant at amp2.
    The smoothness is implemented as a cumulative normal distribution function
    with mean extent1 and standard deviation width1.
    NOTE: The cumulative normal distribution is not equal to 1 at t=0, so if
    the mean is small and/or the width is large, this will mean the IRF will
    start at a smaller value than amp1.
    This function calculates the convolution of a sampled input function with
    the impulse response function.
    The convolution is evaluated at the time points given in t_out and
    returned as a list.

    Arguments:
    t_in    --  The time points of the input function samples.
    in_func --  The input function samples.
    t_out   --  The time points where the model should be evaluated.
    amp1    --  The value of the IRF at t=0
    extent1 --  The midpoint of the transition from amp1 to amp2
    width1  --  The width of the transition between amp1 and amp2
    amp2    --  The value of the IRF at t=inf.

    Return value:
    A list containing the modeled values at each time point.
    """

    res = np.zeros_like(t_out)

    for i in range(0, res.size):
        # For each time point the integrand is integrated.

        # Get current time point
        ti = t_out[i]

        # The integrand is infunc(tau) * IRF(ti-tau)
        def integrand(tau: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return (np.interp(tau, t_in, in_func, left=0.0) *
                    irf_normconst(ti - tau, **kwargs))

        # We integrate each step of the input function separately
        j = 0
        while t_in[j + 1] < ti:
            y = scipy.integrate.quad(integrand, t_in[j], t_in[j + 1])
            res[i] += y[0]
            j = j + 1

        # add last step from t[j] to ti
        y = scipy.integrate.quad(integrand, t_in[j], ti)
        res[i] += y[0]

    return res
