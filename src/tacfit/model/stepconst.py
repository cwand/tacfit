import numpy.typing as npt
import numpy as np


def irf_stepconst(
        t: npt.NDArray[np.float64],
        **kwargs: float) -> npt.NDArray[np.float64]:
    """Computes the IRF of the stepconst model, where the impulse response
    function is assumed to be a step function followed by a constant.
    The step function has value amp1 on the interval
    [0, extent1), followed by the value amp2 on the interval
    [extent1, infinity).

    Arguments:
    t       --  Time points where the IRF should be computed.
    amp1    --  The amplitude of the step function on [0, extent1).
    extent1 --  The length of the first step function.
    amp2    --  The value of the response function on [extent1, infinity).

    Return value:
    A list containing the IRF values at each time point.
    """

    amp1 = kwargs['amp1']
    amp2 = kwargs['amp2']
    ext1 = kwargs['extent1']

    tt = np.atleast_1d(t)
    res = amp2 * np.ones_like(tt)
    for i in range(len(tt)):
        if tt[i] < ext1:
            res[i] = amp1
    return res
