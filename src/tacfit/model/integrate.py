import numpy as np
import numpy.typing as npt
from typing import Callable
import scipy
import tacfit


def model(
        t_in: npt.NDArray[np.float64],
        in_func: npt.NDArray[np.float64],
        t_out: npt.NDArray[np.float64],
        irf: Callable[[npt.NDArray[np.float64],
                       dict[str, float]],
                      npt.NDArray[np.float64]],
        **kwargs: float) -> npt.NDArray[np.float64]:

    """Models a given IRF using a set of parameters and an input function.
    The modeling is performed by calculating the convolution between the input
    function and the impulse response function. The convolution is computed
    for all time points in t_out.

    The keyword arguments passed to this function are passed on to the IRF.
    The exception is the keyword argument '_delay', which shifts the input
    function by the given amount before modeling to account for a delay
    between the input function samples and the response from the tissue owing
    to different measurement locations.

    Parameters:
        t_in    --  The measurement times of the input function
        in_func --  The measured input function
        t_out   --  The time points where the convolution should be computed
        irf     --  The Impulse Response Function. The first positional
                    argument of the IRF is the time points where the IRF
                    should be evaluated. Parameters of the IRF are passed as
                    keyword-arguments.
        kwargs  --  Keyword arguments are passed on to the IRF. The kwarg
                    '_delay' will shift the input time before modeling.

    Returns:
        A numpy-array with the modeled values for each time point in t_out.
    """

    # Prepare result array
    res = np.zeros_like(t_out)

    # Check if delay is a model parameter. In that case, shift the input time
    t_d = 0.0
    if '_delay' in kwargs:
        t_d = kwargs['_delay']
    corr_input_time, corr_input_data = (
        tacfit.create_corrected_input_function(t_in, in_func, t_d))

    for i in range(0, res.size):

        # Get current time point
        ti = t_out[i]

        # The integrand is infunc(tau) * IRF(ti-tau)
        def integrand(tau: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return (np.interp(tau,
                              corr_input_time,
                              corr_input_data,
                              left=0.0) *
                    irf(ti - tau, **kwargs))  # type: ignore

        # We integrate each step of the input function separately
        j = 0
        while corr_input_time[j+1] < ti:
            y = scipy.integrate.quad(integrand,
                                     corr_input_time[j], corr_input_time[j+1])
            res[i] += y[0]
            j = j + 1

        # add last step from t[j] to ti
        y = scipy.integrate.quad(integrand, corr_input_time[j], ti)
        res[i] += y[0]

    return res
