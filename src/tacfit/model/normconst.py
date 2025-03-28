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
    return amp2 + (amp1 - amp2) * (1.0 - cdf)

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

    # Unpack parameters
    ext1 = kwargs['extent1']
    amp1 = kwargs['amp1']
    wid1 = kwargs['width1']
    amp2 = kwargs['amp2']


    for i in range(0, res.size):
        # For each time point the integrand is integrated.

        # Get current time point
        ti = t_out[i]

        # Get time points of input function between t=0 and t=ti
        t_in_cut = [0.0]
        t_in_cut = np.append(t_in_cut, t_in[t_in < ti])
        t_in_cut = np.append(t_in_cut, ti)

        # Get input function samples between t=0 and t=ti
        f_in_cut = [0.0]
        f_in_cut = np.append(f_in_cut, in_func[t_in < ti])
        f_in_cut = np.append(f_in_cut, np.interp(ti, t_in, in_func))

        # IRF needs to be evaluated at ti - tau, where tau is the time
        # points of the input function samples.
        # This interpolates the IRF between these points, which creates an
        # error...

        # Normcdf argument:
        irf_t = ((ti - t_in_cut) - ext1)/(math.sqrt(2.0) * wid1)

        # Calculate normcdf using error function (seems to be much quicker)
        cdf = 0.5 * (1.0 + scipy.special.erf(irf_t))

        # Calculate input response function
        irf = (amp2 + (amp1 - amp2) * (1.0 - cdf))

        # Compute convolution
        res[i] = scipy.integrate.trapezoid(irf*f_in_cut, t_in_cut)

    return res
