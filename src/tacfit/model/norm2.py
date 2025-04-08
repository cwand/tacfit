import math

import numpy as np
import numpy.typing as npt
import scipy


def irf_norm2(
        t: npt.NDArray[np.float64],
        **kwargs: float) -> npt.NDArray[np.float64]:
    """Computes the IRF of the norm2 model, where the impulse response
    function starts at the value amp1, transitions smoothly to a new value
    amp2 around the time point extent1, and then transitions smoothly to zero
    around time extent2.
    The smoothness is implemented as a cumulative normal distribution function
    with mean extent1(extent2) and standard deviation width1(width2).
    NOTE: The cumulative normal distribution is not equal to 1 at t=0, so if
    the mean is small and/or the width is large, this will mean the IRF will
    start at a smaller value than amp1.

    Arguments:
    t       --  Time points where the IRF should be computed.
    amp1    --  The value of the IRF at t=0
    extent1 --  The midpoint of the transition from amp1 to amp2
    width1  --  The width of the transition between amp1 and amp2
    amp2    --  The value of the IRF after the first transition.
    extent2 --  The midpoint of the transition from amp2 to 0.
    width2  --  The width of the transition between amp2 and 0

    Return value:
    A list containing the IRF values at each time point.
    """
    ext1 = kwargs['extent1']
    amp1 = kwargs['amp1']
    wid1 = kwargs['width1']
    amp2 = kwargs['amp2']
    ext2 = kwargs['extent2']
    wid2 = kwargs['width2']

    # Calculate normcdf using error function (seems to be much quicker)
    tt1 = (t - ext1) / (math.sqrt(2.0) * wid1)
    cdf1 = 0.5 * (1.0 + scipy.special.erf(tt1))

    tt2 = (t - ext2) / (math.sqrt(2.0) * wid2)
    cdf2 = 0.5 * (1.0 + scipy.special.erf(tt2))

    # Calculate input response function
    return np.array(amp2 * (1.0 - cdf2) + (amp1 - amp2) * (1.0 - cdf1))
