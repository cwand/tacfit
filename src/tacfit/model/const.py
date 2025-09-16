import numpy as np
import numpy.typing as npt


def irf_const(
        t: npt.NDArray[np.float64],
        **kwargs: float) -> npt.NDArray[np.float64]:
    """Computes the IRF of the const model, where the impulse response
    function is simply a constant, i.e. the tracer accumulates in the tissue.

    Arguments:
    t       --  Time points where the IRF should be computed.
    amp    --  The value of the constant IRF

    Return value:
    A list containing the IRF values at each time point.
    """

    amp = kwargs['amp']

    # Calculate input response function
    return amp * np.ones_like(t)
