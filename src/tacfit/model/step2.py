import numpy as np
import numpy.typing as npt
import scipy


def irf_step2(
        t: npt.NDArray[np.float64],
        **kwargs: float) -> npt.NDArray[np.float64]:
    """Computes the IRF of the step2 model, where the input response function
    is assumed to be a 2-step function.
    The step function has value amp1 on the interval
    [0, extent1), value amp2 on the interval [extent1, extent2) and value 0 on
    the interval [extent2, infinity).

    Arguments:
    t       --  Time points where the IRF should be computed.
    amp1    --  The amplitude of the step function on [0, extent1).
    extent1 --  The length of the first step function.
    amp2    --  The amplitude of the step function on [0, extent2).
    extent2 --  The length of the second step function.

    Return value:
    A list containing the IRF values at each time point.
    """

    amp1 = kwargs['amp1']
    amp2 = kwargs['amp2']
    ext1 = kwargs['extent1']
    ext2 = kwargs['extent2']

    tt = np.array(t, ndmin=1)

    # Calculate normcdf using error function (seems to be much quicker)
    res = np.zeros_like(tt)
    for i in range(len(tt)):
        if tt[i] < ext1:
            res[i] = amp1
        elif tt[i] < ext2:
            res[i] = amp2
    return res
